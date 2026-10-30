"""Segment point clouds with an exported PTv3 model: ONNX Runtime + NumPy, no PyTorch.

    python -m ptv3.runtime.onnx_infer --model ptv3_s3dis.onnx --data-root data/s3dis --area Area_5

Works with the CUDA, DirectML (Windows) or CPU execution provider. Fragments of one scene are
batched up to --max-points points per run: that number sets the GPU memory used (see docs/DEPLOY.md).
"""
import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort

from ..data import s3dis
from ..data import transforms as T
from ..export.prep import collate, prepare
from ..metrics import confusion, scores

PROVIDERS = {
    "cuda": [("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})],
    "directml": ["DmlExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}


def process_gpu_gb():
    """Device memory this process holds (nvidia-smi; ONNX Runtime's arena keeps its peak). 0 without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
    except OSError:
        return 0.0
    return sum(float(m) for p, m in (l.split(",") for l in out.strip().splitlines()) if int(p) == os.getpid()) / 1024


class PeakGPUMemory:
    """Sample this process's device memory (nvidia-smi) in a thread; .peak in GB. For reports only."""

    def __init__(self, every=0.05):
        self.every, self.peak, self._stop = every, 0.0, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, process_gpu_gb())
            self._stop.wait(self.every)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def preload_cuda_libraries():
    """Make cuDNN / cuBLAS from the nvidia-* pip packages visible to ONNX Runtime's CUDA provider.

    Without PyTorch nothing else loads them, and the CUDA provider fails with "libcudnn.so.9: cannot
    open shared object file". ONNX Runtime >= 1.21 does this itself (ort.preload_dlls).
    """
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
        return
    import ctypes
    import glob
    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    root = list(spec.submodule_search_locations)[0]
    for lib in ("cuda_runtime", "cublas", "cufft", "curand", "cudnn"):
        for path in sorted(glob.glob(os.path.join(root, lib, "lib", "lib*.so*"))):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def create_session(path, provider="auto", threads=0):
    order = ["cuda", "directml", "cpu"] if provider == "auto" else [provider]
    if "cuda" in order:
        preload_cuda_libraries()
    opts = ort.SessionOptions()
    # Memory patterns pre-plan one block for a run's whole lifetime from the shapes of the previous
    # run; with input sizes that change every run they cost ~40 % more device memory (docs/DEPLOY.md).
    opts.enable_mem_pattern = False
    if threads:
        opts.intra_op_num_threads = threads
    for name in order:
        try:
            session = ort.InferenceSession(path, opts, providers=PROVIDERS[name])
        except Exception as exc:                                    # provider library missing, old driver, ...
            print(f"could not create a {name} session: {exc}")
            continue
        wanted = PROVIDERS[name][0]
        wanted = wanted if isinstance(wanted, str) else wanted[0]
        if session.get_providers()[0] == wanted:
            return session
    raise RuntimeError("no execution provider could load the model")


class Segmenter:
    def __init__(self, model_path, provider="auto", max_points=400_000, workers=2, lookahead=1, margin=1.0,
                 shrink_arena=True):
        self.session = create_session(model_path, provider)
        # Return the arena's unused blocks to the device after every run. Without it the arena keeps
        # blocks from every earlier run size and grows past what one run needs (Area 5 with slabs of
        # 400k points: 13 GB held vs a 5.3 GB peak per run), for ~20 % more time per run.
        self.run_options = ort.RunOptions()
        if shrink_arena and self.session.get_providers()[0] == "CUDAExecutionProvider":
            self.run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
        self.cfg = json.loads(self.session.get_modelmeta().custom_metadata_map["ptv3_config"])
        self.inputs = [i.name for i in self.session.get_inputs()]
        self.max_points = max_points
        self.workers, self.lookahead, self.margin = workers, lookahead, margin

    def batches(self, fragments):
        """Group consecutive fragments into runs of at most max_points points (one fragment at least)."""
        out, i = [], 0
        while i < len(fragments):
            j, n = i, 0
            while j < len(fragments) and (j == i or n + len(fragments[j]["feat"]) <= self.max_points):
                n += len(fragments[j]["feat"])
                j += 1
            out.append(fragments[i:j])
            i = j
        return out

    def tables(self, batch):
        return collate([prepare(f["grid_coord"], f["feat"], self.cfg) for f in batch], self.cfg)

    def logits(self, fragments):
        """fragments: list of dict(grid_coord, feat) of the SAME scene and augmentation -> list of (n, C).

        Test fragments of one augmentation (GridSample "test": the i-th point of every voxel) all
        cover the same voxels in the same order, so their index tables are identical: they are built
        once per batch size and only the features change. Otherwise the tables of the next batches
        are prepared in worker threads while the current batch runs (NumPy's sorts and searches and
        ONNX Runtime's run() release the GIL); at most `lookahead` batches ahead, to bound the RAM.
        """
        batches = self.batches(fragments)
        grid = fragments[0]["grid_coord"]
        shared = all(f["grid_coord"] is grid or np.array_equal(f["grid_coord"], grid) for f in fragments[1:])
        out = []
        if shared:
            built = {}
            for batch in batches:
                if len(batch) not in built:
                    built[len(batch)] = self.tables(batch)
                tables = dict(built[len(batch)])
                tables["feat0"] = np.concatenate([f["feat"] for f in batch]).astype(np.float32)
                out += self.run(tables, batch)
            return out
        pending = []
        with ThreadPoolExecutor(self.workers) as pool:
            for k, batch in enumerate(batches):
                while len(pending) <= self.lookahead and k + len(pending) < len(batches):
                    pending.append(pool.submit(self.tables, batches[k + len(pending)]))
                out += self.run(pending.pop(0).result(), batch)
        return out

    def run(self, tables, batch):
        result = self.session.run(None, {name: tables[name] for name in self.inputs}, self.run_options)[0]
        return np.split(result, np.cumsum([len(f["feat"]) for f in batch])[:-1])

    def slabs(self, grid, grid_size):
        """Split a fragment larger than max_points into slabs along its longer horizontal axis.

        Returns (rows, core) pairs: `rows` are the slab's points plus a `margin` (metres) of context on
        each side, `core` marks the rows whose prediction is kept. Computed from the voxel grid, which
        every fragment of one augmentation shares, so all of them split the same way.
        """
        n = len(grid)
        if n <= self.max_points:
            return [(np.arange(n), np.ones(n, dtype=bool))]
        axis = int(np.argmax(np.ptp(grid[:, :2], axis=0)))
        x = grid[:, axis]
        m = self.margin / grid_size
        k = int(np.ceil(n / self.max_points))
        while True:                                     # more slabs until every slab + margins fits
            k += 1
            cuts = np.quantile(x, np.linspace(0, 1, k + 1))
            cuts[-1] += 1
            out = []
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                rows = np.nonzero((x >= lo - m) & (x < hi + m))[0]
                out.append((rows, (x[rows] >= lo) & (x[rows] < hi)))
            if max(len(r) for r, _ in out) <= self.max_points or k >= 256:     # 256: margin too wide to fit
                return out

    def segment(self, scene, augmentations=(s3dis.TTA[2],), grid_size=0.02):
        """scene: dict(coord, color, normal) -> (N, C) summed class probabilities."""
        probs = np.zeros((len(scene["coord"]), self.cfg["num_classes"]), dtype=np.float32)
        for aug in augmentations:
            fragments = s3dis.test_fragments(scene, grid_size, [aug])
            for rows, core in self.slabs(fragments[0]["grid_coord"], grid_size):
                parts = [{"grid_coord": f["grid_coord"][rows], "feat": f["feat"][rows]} for f in fragments]
                for frag, logit in zip(fragments, self.logits(parts)):
                    logit = logit[core]
                    e = np.exp(logit - logit.max(1, keepdims=True))
                    probs[frag["index"][rows[core]]] += e / e.sum(1, keepdims=True)   # each point once per fragment
        return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--area", default=s3dis.TEST_AREA)
    ap.add_argument("--provider", default="auto", choices=["auto", "cuda", "directml", "cpu"])
    ap.add_argument("--max-points", type=int, default=400_000)
    ap.add_argument("--margin", type=float, default=1.0,
                    help="metres of context around each slab when a fragment is split (fragments > --max-points)")
    ap.add_argument("--tta", action="store_true", help="the official 10 test-time augmentations")
    ap.add_argument("--limit", type=int, default=None, help="first N scenes only")
    ap.add_argument("--out", default=None, help="write the scores as JSON")
    ap.add_argument("--keep-arena", action="store_true", help="don't shrink ONNX Runtime's GPU arena after each run")
    args = ap.parse_args()

    seg = Segmenter(args.model, args.provider, args.max_points, margin=args.margin, shrink_arena=not args.keep_arena)
    print("providers:", seg.session.get_providers())
    augs = s3dis.TTA if args.tta else [s3dis.TTA[2]]
    conf = np.zeros((seg.cfg["num_classes"],) * 2, dtype=np.int64)
    t0 = time.time()
    base = process_gpu_gb()                  # CUDA context + weights, before any run
    with PeakGPUMemory() as memory:
        for path in s3dis.rooms(args.data_root, args.area)[: args.limit]:
            room = s3dis.load_room(path)
            pred = seg.segment(room, augs).argmax(1)
            conf += confusion(pred, room["segment"].astype(np.int64), seg.cfg["num_classes"])
            print(f"{room['name']:24s} running mIoU {scores(conf)['mIoU']:.4f}  {time.time() - t0:.0f} s", flush=True)
    result = {**scores(conf), "seconds": time.time() - t0, "provider": seg.session.get_providers()[0],
              "max_points": args.max_points, "margin": args.margin, "tta": args.tta,
              "gpu_gb": memory.peak, "gpu_gb_after_load": base}
    print(json.dumps({k: v for k, v in result.items() if k != "iou"}))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1, default=float)


if __name__ == "__main__":
    main()
