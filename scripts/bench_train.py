"""Training step time and peak GPU memory: official PTv3 vs this implementation, on identical batches.

    python scripts/bench_train.py --impl official --official-repo third_party/PointTransformerV3 --data-root data/s3dis
    python scripts/bench_train.py --impl ours --attn-backend triton --conv-backend triton --data-root data/s3dis

A fixed list of micro-batches is drawn once from the S3DIS training pipeline (seeded, with the
official augmentations and mixing) and replayed, so both implementations see the same inputs.
One step = forward + loss (CE + Lovasz) + backward + AdamW, under the given autocast.
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ptv3.checkpoint import build  # noqa: E402
from ptv3.data import s3dis  # noqa: E402
from ptv3.losses import SegLoss  # noqa: E402
from ptv3.presets import FAST, PRESETS  # noqa: E402
from ptv3.train import collate, param_groups  # noqa: E402


def fixed_batches(root, n, micro_batch, mix_prob, seed):
    random.seed(seed)
    np.random.seed(seed)
    data = s3dis.S3DIS(root, loop=1)
    idx = np.random.permutation(len(data))
    return [collate([data[int(i)] for i in idx[k * micro_batch:(k + 1) * micro_batch]], mix_prob) for k in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=["official", "ours"], default="ours")
    ap.add_argument("--official-repo")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--preset", default="s3dis-rpe")
    ap.add_argument("--micro-batch", type=int, default=3)
    ap.add_argument("--batches", type=int, default=8, help="distinct batches, replayed in a loop")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5, help="at least one pass over the distinct batches is added")
    ap.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "none"])
    ap.add_argument("--attn-backend", default="sdpa", choices=["sdpa", "math", "triton", "flash"])
    ap.add_argument("--conv-backend", default="torch", choices=["torch", "triton"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    batches = fixed_batches(args.data_root, args.batches, args.micro_batch, 0.8, args.seed)
    torch.manual_seed(args.seed)
    if args.impl == "official":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from baselines.official import official_segmentor, pointcept_loss
        model, loss_fn = official_segmentor(args.official_repo, PRESETS[args.preset]), pointcept_loss
        fused = False
    else:
        model = build(args.preset, **{**FAST, "attn_backend": args.attn_backend, "conv_backend": args.conv_backend})
        loss_fn, fused = SegLoss(), True
    model = model.cuda().train()
    opt = torch.optim.AdamW(param_groups(model, 0.006, 0.0006), lr=0.006, weight_decay=0.05, fused=fused)
    amp = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": None}[args.amp]
    scaler = torch.amp.GradScaler(enabled=args.amp == "fp16")

    args.warmup = max(args.warmup, args.batches)       # every input shape seen once: Triton compiles out of the timing
    times, points = [], []
    for step in range(args.warmup + args.steps):
        if step == args.warmup:
            torch.cuda.reset_peak_memory_stats()
        batch = {k: v.cuda(non_blocking=True) for k, v in batches[step % len(batches)].items()}
        torch.cuda.synchronize()
        t = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp or torch.float32, enabled=amp is not None):
            loss = loss_fn(model({k: batch[k] for k in ("coord", "grid_coord", "feat", "offset")}), batch["segment"])
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        if step >= args.warmup:
            times.append(time.time() - t)
            points.append(len(batch["coord"]))
    res = {"impl": args.impl, "attn": args.attn_backend, "conv": args.conv_backend, "amp": args.amp,
           "micro_batch": args.micro_batch, "ms_per_step": 1000 * float(np.mean(times)),
           "points_per_s": float(np.sum(points) / np.sum(times)), "mean_points": float(np.mean(points)),
           "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9}
    print(json.dumps(res))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
