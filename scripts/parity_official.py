"""Compare this implementation with the official PTv3 on real S3DIS fragments, same weights, same seed.

    python scripts/parity_official.py --official-repo third_party/PointTransformerV3 \\
        --checkpoint weights/s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth --data-root data/s3dis --rooms 5

For each room: one fragment (scale 1.0, no flip), the official model (spconv + its attention) and
each backend here, with torch.manual_seed reset before every forward so the random order shuffles
upstream applies at test time are the same on both sides, and the UPSTREAM preset (current upstream
code). Prints the largest logit difference and the share of points with the same predicted class.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from baselines.official import official_segmentor  # noqa: E402
from ptv3.checkpoint import load, load_state_dict  # noqa: E402
from ptv3.data import s3dis  # noqa: E402
from ptv3.presets import OFFICIAL, PRESETS, UPSTREAM  # noqa: E402

BACKENDS = {"math + torch conv": dict(attn_backend="math", conv_backend="torch"),
            "sdpa + torch conv": dict(attn_backend="sdpa", conv_backend="torch"),
            "triton + triton conv": dict(attn_backend="triton", conv_backend="triton")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--official-repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--preset", default="s3dis-rpe")
    ap.add_argument("--rooms", type=int, default=5)
    ap.add_argument("--mode", default="upstream", choices=["upstream", "official"],
                    help="upstream: compare with the current official code; official: with its Dec 2023 release "
                         "(git checkout 37a3ddc), the code the checkpoints were trained with")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    official = official_segmentor(args.official_repo, PRESETS[args.preset], load_state_dict(args.checkpoint)).cuda().eval()
    ours = {name: load(args.preset, args.checkpoint, **{**(UPSTREAM if args.mode == "upstream" else OFFICIAL), **o}).cuda() for name, o in BACKENDS.items()}
    rows = []
    for path in s3dis.rooms(args.data_root, s3dis.TEST_AREA)[: args.rooms]:
        frag = s3dis.test_fragments(s3dis.load_room(path), augmentations=[s3dis.TTA[2]])[0]
        inp = {k: torch.from_numpy(frag[k]).cuda() for k in ("coord", "grid_coord", "feat")}
        inp["offset"] = torch.tensor([len(frag["coord"])], device="cuda")
        with torch.no_grad():
            torch.manual_seed(0)
            ref = official({k: v.clone() for k, v in inp.items()}).float()
            for name, model in ours.items():
                torch.manual_seed(0)
                out = model({k: v.clone() for k, v in inp.items()}).float()
                rec = {"room": os.path.basename(path), "points": len(frag["coord"]), "backend": name,
                       "max_abs_diff": float((out - ref).abs().max()),
                       "same_class": float((out.argmax(1) == ref.argmax(1)).float().mean())}
                rows.append(rec)
                print(json.dumps(rec), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
