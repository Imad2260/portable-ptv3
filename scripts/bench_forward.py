"""Forward latency and peak GPU memory against input size, for each implementation.

    python scripts/bench_forward.py --data-root data/s3dis --checkpoint ... --impls official,ours-sdpa,ours-triton,onnx \\
        --official-repo third_party/PointTransformerV3 --onnx ptv3_s3dis.onnx --out results/forward.jsonl

Inputs are real S3DIS fragments (one point per 2 cm voxel), concatenated room by room up to each
target size, so attention patches and neighbour maps look like the real workload. PyTorch memory
is the allocator's peak; ONNX Runtime memory is the device memory its process holds after the run
(its arena keeps the peak) beyond what it held once the session was loaded, read per process with nvidia-smi.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ptv3.checkpoint import load, load_state_dict  # noqa: E402
from ptv3.data import s3dis  # noqa: E402
from ptv3.presets import FAST, PRESETS  # noqa: E402


def fragments_up_to(root, sizes):
    """For every target size, one-scene-per-sample batches whose total point count reaches it."""
    paths = s3dis.rooms(root, s3dis.TEST_AREA)
    pool, total = [], 0
    for p in paths:
        pool.append(s3dis.test_fragments(s3dis.load_room(p), augmentations=[s3dis.TTA[2]])[0])
        total += len(pool[-1]["coord"])
        if total >= max(sizes):
            break
    out = {}
    for size in sizes:
        chosen, n = [], 0
        for f in pool:
            take = min(len(f["coord"]), size - n)
            if take <= 0:
                break
            # a spatial crop (the points nearest the centre), not a prefix: fragment rows are in hash
            # order, so a prefix would be a random, unrealistically sparse subset of the voxels
            keep = np.argsort(np.square(f["coord"] - f["coord"].mean(0)).sum(1))[:take]
            chosen.append({k: v[keep] for k, v in f.items() if k in ("coord", "grid_coord", "feat")})
            n += take
        out[size] = chosen
    return out


def process_gpu_gb():
    """Device memory held by this process (nvidia-smi), so other processes on the GPU don't count."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return sum(float(m) for p, m in (l.split(",") for l in out.strip().splitlines()) if int(p) == os.getpid()) / 1024


def torch_batch(samples):
    b = {k: torch.from_numpy(np.concatenate([s[k] for s in samples])).cuda() for k in ("coord", "grid_coord", "feat")}
    b["offset"] = torch.tensor(np.cumsum([len(s["coord"]) for s in samples])).cuda()
    return b


def time_torch(model, samples, amp, repeats):
    times = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    for _ in range(repeats + 1):
        batch = torch_batch(samples)
        torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=amp or torch.float32, enabled=amp is not None):
            model(batch)
        torch.cuda.synchronize()
        times.append(time.time() - t)
    return 1000 * float(np.median(times[1:])), (torch.cuda.max_memory_allocated() - base) / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--preset", default="s3dis-rpe")
    ap.add_argument("--impls", default="ours-sdpa,ours-triton")
    ap.add_argument("--official-repo")
    ap.add_argument("--onnx")
    ap.add_argument("--onnx-label", default="onnx", help="name of the ONNX row in the output (e.g. onnx-fused)")
    ap.add_argument("--sizes", default="50000,100000,200000,400000,800000,1600000")
    ap.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    inputs = fragments_up_to(args.data_root, sizes)
    amp = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]
    for impl in args.impls.split(","):
        if impl == "onnx":
            from ptv3.export.prep import collate, prepare
            from ptv3.runtime.onnx_infer import create_session
            session = create_session(args.onnx, "cuda")
            cfg = json.loads(session.get_modelmeta().custom_metadata_map["ptv3_config"])
            names = [i.name for i in session.get_inputs()]
            base_gb = process_gpu_gb()
        elif impl == "official":
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from baselines.official import official_segmentor
            model = official_segmentor(args.official_repo, PRESETS[args.preset], load_state_dict(args.checkpoint)).cuda().eval()
        else:
            attn, conv = {"ours-sdpa": ("sdpa", "torch"), "ours-triton": ("triton", "triton"),
                          "ours-math": ("math", "torch")}[impl]
            model = load(args.preset, args.checkpoint, **{**FAST, "attn_backend": attn, "conv_backend": conv}).cuda()
        for size in sizes:
            samples = inputs[size]
            rec = {"impl": args.onnx_label if impl == "onnx" else impl, "points": sum(len(s["coord"]) for s in samples), "amp": args.amp}
            try:
                if impl == "onnx":
                    t = time.time()
                    tables = collate([prepare(s["grid_coord"], s["feat"], cfg) for s in samples], cfg)
                    rec["prep_ms"] = 1000 * (time.time() - t)
                    times = []
                    for _ in range(args.repeats + 1):
                        t = time.time()
                        session.run(None, {k: tables[k] for k in names})
                        times.append(time.time() - t)
                    rec["ms"] = 1000 * float(np.median(times[1:]))
                    rec["peak_gpu_gb"] = process_gpu_gb() - base_gb
                else:
                    rec["ms"], rec["peak_gpu_gb"] = time_torch(model, samples, amp, args.repeats)
            except Exception as exc:                  # out of memory (torch or ONNX Runtime)
                rec["error"] = str(exc).splitlines()[0][:200]
                torch.cuda.empty_cache()
            print(json.dumps(rec), flush=True)
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        if impl != "onnx":
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
