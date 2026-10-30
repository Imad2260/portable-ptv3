"""Evaluate the official PTv3 or this implementation on S3DIS Area 5 with the same protocol code.

    # upstream model, upstream protocol (one fragment per forward pass, fp32)
    python scripts/evaluate.py --impl official --official-repo third_party/PointTransformerV3 \\
        --checkpoint weights/s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth --data-root data/s3dis \\
        --max-points 0 --out results/official.json

    # this implementation, same weights
    python scripts/evaluate.py --impl ours --checkpoint ... --data-root data/s3dis \\
        --attn-backend triton --conv-backend triton --max-points 1500000 --amp bf16 --out results/ours.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402

from ptv3.checkpoint import load, load_state_dict  # noqa: E402
from ptv3.evaluate import evaluate  # noqa: E402
from ptv3.presets import FAST, OFFICIAL, PRESETS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=["official", "ours"], default="ours")
    ap.add_argument("--official-repo", help="clone of github.com/Pointcept/PointTransformerV3 (--impl official)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--preset", default="s3dis-rpe", choices=sorted(PRESETS))
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--protocol", default="official", choices=["official", "single", "fast"])
    ap.add_argument("--max-points", type=int, default=1_500_000, help="points per forward pass; 0 = one fragment")
    ap.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    ap.add_argument("--mode", default="fast", choices=["fast", "official"],
                    help="ours: deterministic eval (fast) or upstream's random order shuffling (official)")
    ap.add_argument("--attn-backend", default="sdpa", choices=["sdpa", "math", "triton", "flash"])
    ap.add_argument("--conv-backend", default="torch", choices=["torch", "triton"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--color-norm", default="pm1", choices=["pm1", "unit"],
                    help="pm1: c/127.5-1, what the official checkpoints were trained with; unit: c/255, current Pointcept")
    ap.add_argument("--mode-cpe", default=None, choices=["mlp_branch", "residual"], help="override cpe_input (ours)")
    ap.add_argument("--order", default=None,
                    help="comma-separated curve order for --mode fast, e.g. hilbert,hilbert-trans,z,z-trans")
    ap.add_argument("--limit", type=int, default=None, help="first N rooms only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from ptv3.data import s3dis
    s3dis.COLOR_NORM = args.color_norm
    if args.impl == "official":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from baselines.official import official_segmentor
        model = official_segmentor(args.official_repo, PRESETS[args.preset], load_state_dict(args.checkpoint))
    else:
        options = dict(FAST if args.mode == "fast" else OFFICIAL)
        options.update(attn_backend=args.attn_backend, conv_backend=args.conv_backend)
        if args.mode_cpe:
            options.update(cpe_input=args.mode_cpe)
        if args.order:
            options.update(order=tuple(args.order.split(",")))
        model = load(args.preset, args.checkpoint, **options)
    model = model.cuda().eval()
    amp = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]
    res = evaluate(model, args.data_root, protocol=args.protocol, num_classes=PRESETS[args.preset]["num_classes"],
                   max_points=args.max_points, amp_dtype=amp, workers=args.workers, limit=args.limit)
    res["args"] = vars(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("mIoU", "mAcc", "allAcc", "seconds", "forward_seconds", "peak_gpu_gb")}))


if __name__ == "__main__":
    main()
