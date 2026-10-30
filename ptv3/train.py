"""Train PTv3 on S3DIS with the official recipe (configs/s3dis/semseg-pt-v3m1-0-rpe.py of Pointcept).

    python -m ptv3.train --data-root data/s3dis --out runs/s3dis

Upstream trained on 4 GPUs with 3 samples each (12 per step, BatchNorm statistics per GPU). On one
GPU the same step is 4 micro-batches of 3 samples with accumulated gradients (--micro-batch 3).
Everything else follows the config: AdamW (lr 6e-3, 6e-4 for attention blocks, wd 0.05),
OneCycle (5 % warm-up, cosine), fp16 autocast with loss scaling, Mix3D-style mixing with p = 0.8,
100 epochs of 30 passes over the training rooms.
"""
import argparse
import functools
import json
import math
import os
import random
import time

import numpy as np
import torch

from . import presets
from .checkpoint import build, load_state_dict
from .data import s3dis
from .evaluate import evaluate
from .losses import SegLoss


def collate(samples, mix_prob=0.0):
    """Concatenate samples (Pointcept's point_collate_fn).

    With probability `mix_prob` every two consecutive samples are merged into one by dropping the
    offset between them. Their voxel grids are not recomputed, so voxels of the two scenes can
    coincide; upstream does the same.
    """
    batch = {k: torch.from_numpy(np.concatenate([s[k] for s in samples])) for k in ("coord", "grid_coord", "feat", "segment")}
    offset = torch.tensor(np.cumsum([len(s["coord"]) for s in samples]))
    if random.random() < mix_prob:
        offset = torch.cat([offset[1:-1:2], offset[-1:]])
    batch["offset"] = offset
    return batch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def param_groups(model, lr, block_lr, keyword="block"):
    """Pointcept's param_dicts: parameters whose name contains `keyword` get their own learning rate."""
    base = [p for n, p in model.named_parameters() if keyword not in n]
    blocks = [p for n, p in model.named_parameters() if keyword in n]
    return [{"params": base, "lr": lr}, {"params": blocks, "lr": block_lr}]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="runs/s3dis")
    ap.add_argument("--preset", default="s3dis-rpe", choices=sorted(presets.PRESETS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--loop", type=int, default=30, help="passes over the training rooms per epoch")
    ap.add_argument("--batch-size", type=int, default=12, help="samples per optimizer step")
    ap.add_argument("--micro-batch", type=int, default=3, help="samples per forward pass")
    ap.add_argument("--lr", type=float, default=0.006)
    ap.add_argument("--block-lr", type=float, default=0.0006)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--mix-prob", type=float, default=0.8)
    ap.add_argument("--amp", default="fp16", choices=["fp16", "bf16", "none"])
    ap.add_argument("--attn-backend", default="sdpa", choices=["sdpa", "math", "flash", "triton"])
    ap.add_argument("--patch-mode", default=None, choices=["fixed", "min"])
    ap.add_argument("--unpool-conv-input", default="skip", choices=["skip", "sum"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=25326354)
    ap.add_argument("--eval-every", type=int, default=10, help="epochs between validations (0: never)")
    ap.add_argument("--eval-protocol", default="fast", choices=["fast", "single", "official"])
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (benchmarks)")
    ap.add_argument("--resume", default="")
    ap.add_argument("--init", default="", help="start from these weights (e.g. an official checkpoint)")
    args = ap.parse_args(argv)
    assert args.batch_size % args.micro_batch == 0
    accum = args.batch_size // args.micro_batch
    os.makedirs(args.out, exist_ok=True)
    seed_everything(args.seed)
    device = "cuda"

    model = build(args.preset, attn_backend=args.attn_backend, patch_mode=args.patch_mode,
                  unpool_conv_input=args.unpool_conv_input, shuffle_at_eval=False).to(device)
    if args.init:
        model.load_state_dict(load_state_dict(args.init))
    dataset = s3dis.S3DIS(args.data_root, loop=args.loop)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.micro_batch, shuffle=True, drop_last=True, num_workers=args.workers,
        collate_fn=functools.partial(collate, mix_prob=args.mix_prob), persistent_workers=args.workers > 0,
        pin_memory=True,
    )  # PyTorch seeds random, NumPy and torch differently in every worker
    steps_per_epoch = len(loader) // accum
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(param_groups(model, args.lr, args.block_lr), lr=args.lr,
                                  weight_decay=args.weight_decay, fused=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr, args.block_lr], total_steps=total_steps, pct_start=0.05,
        anneal_strategy="cos", div_factor=10.0, final_div_factor=1000.0)
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": None}[args.amp]
    scaler = torch.amp.GradScaler(enabled=args.amp == "fp16")
    criterion = SegLoss()

    start_epoch, step, best = 0, 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, step, best = ck["epoch"], ck["step"], ck.get("best", -1.0)

    log_file = open(os.path.join(args.out, "log.jsonl"), "a")

    def save(name, epoch):
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "step": step, "best": best, "args": vars(args)},
                   os.path.join(args.out, name))

    print(f"{len(dataset)} samples/epoch, {steps_per_epoch} steps/epoch x {args.epochs} epochs = {total_steps} steps, "
          f"{accum} micro-batches of {args.micro_batch} per step")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, seen_points, losses = time.time(), 0, []
        torch.cuda.reset_peak_memory_stats()
        it = iter(loader)
        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accum):
                batch = {k: v.to(device, non_blocking=True) for k, v in next(it).items()}
                seen_points += len(batch["coord"])
                with torch.autocast(device_type="cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                    loss = criterion(model(batch), batch["segment"]) / accum
                scaler.scale(loss).backward()
                losses.append(loss.item() * accum)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scale <= scaler.get_scale():         # upstream skips the scheduler step when fp16 overflowed
                scheduler.step()
            step += 1
            if step % 50 == 0:
                el = time.time() - t0
                rec = {"epoch": epoch + 1, "step": step, "loss": float(np.mean(losses[-50 * accum:])),
                       "lr": scheduler.get_last_lr()[0], "s_per_step": el / (step - epoch * steps_per_epoch),
                       "points_per_s": seen_points / el, "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
                print(json.dumps(rec), flush=True)
                log_file.write(json.dumps(rec) + "\n")
                log_file.flush()
            if args.max_steps and step >= args.max_steps:
                return
        save("last.pth", epoch + 1)
        if args.eval_every and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs):
            res = evaluate(model, args.data_root, protocol=args.eval_protocol, log=lambda *_: None, workers=8)
            rec = {"epoch": epoch + 1, "step": step, "val_mIoU": res["mIoU"], "val_allAcc": res["allAcc"]}
            print(json.dumps(rec), flush=True)
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()
            if res["mIoU"] > best:
                best = res["mIoU"]
                save("best.pth", epoch + 1)


if __name__ == "__main__":
    main()
