"""Turn the benchmark outputs (results/*.json, *.jsonl) into docs/RESULTS.md.

    python scripts/make_results.py --results results --out docs/RESULTS.md
"""
import argparse
import json
import os


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="docs/RESULTS.md")
    args = ap.parse_args()
    R = lambda name: load(os.path.join(args.results, name))   # noqa: E731
    md = ["# Results", "",
          "All numbers measured by the scripts in `scripts/`, on one GPU with nothing else running; raw",
          "outputs are not kept in the repository. Weights: the official PTv3 S3DIS checkpoint",
          "(`Pointcept/PointTransformerV3`, `s3dis-semseg-pt-v3m1-0-rpe`). Data: S3DIS Area 5 (68 rooms,",
          "78.7 M points), Pointcept's preprocessed copy.", ""]

    def eval_table(files, ref):
        base = R(ref)
        rows = []
        for name, f in files:
            r = R(f)
            if not r:
                continue
            row = [name, f"{100 * r['mIoU']:.2f}", f"{100 * r['allAcc']:.2f}", f"{r['seconds'] / 60:.1f} min"]
            row.append(f"{base['seconds'] / r['seconds']:.1f}x" if base else "")
            row.append(f"{r['peak_gpu_gb']:.1f} GB" if "peak_gpu_gb" in r else f"{r.get('gpu_gb', 0):.1f} GB (process)")
            row.append(f"{r['forward_seconds'] / 60:.1f} min" if "forward_seconds" in r else "")
            rows.append(row)
        return table(["Run", "mIoU %", "allAcc %", "Wall time", "Speed-up", "Peak GPU memory", "GPU forward time"], rows) if rows else None

    tta = eval_table([("official code (Dec 2023 release, the checkpoint's code)", "eval_official2023.json"),
                      ("this repo, Triton kernels, fp32", "eval_ours_fast.json"),
                      ("this repo, Triton kernels, bf16", "eval_ours_bf16.json")], "eval_official2023.json")
    if tta:
        md += ["## Accuracy and evaluation time, S3DIS Area 5", "",
               "Official protocol: every point covered by fragments, 10 test-time augmentations (5 scales x flip),",
               "probabilities summed (Pointcept's `SemSegTester`). The official training log of this checkpoint",
               "reports 73.58 mIoU. Wall time includes data loading; this repo batches fragments up to 1.5 M points",
               "per forward pass, the official code runs one fragment per pass.", "", tta, ""]
    single = eval_table([("official code (Dec 2023 release)", "eval_official2023_single.json"),
                         ("this repo, Triton kernels, fp32", "eval_ours_single.json"),
                         ("ONNX, standard operators, ONNX Runtime CUDA, no PyTorch (no split, arena kept)", "eval_onnx_single.json"),
                         ("ONNX, fused fp16 attention (no split, arena kept)", "eval_onnx_fused_single.json"),
                         ("ONNX, fused, slabs of <= 400k points (1 m context), arena shrunk: the runtime default", "eval_onnx_fused_single_400000.json"),
                         ("official code today (2025 colour normalization)", "eval_officialnow_unit_single.json"),
                         ("official code today, 2023 colour normalization", "eval_officialnow_pm1_single.json")],
                        "eval_official2023_single.json")
    if single:
        md += ["## Single pass (no test-time augmentation)", "",
               "One augmentation (scale 1, no flip), every point covered. ONNX rows: NumPy preparation + ONNX",
               "Runtime only; GPU memory is what the process holds (ONNX Runtime keeps its peak). Without",
               "splitting, the largest room runs as one fragment, whatever --max-points says. The last two rows run the same",
               "checkpoint with the current official code: see the README, \"The official checkpoints no longer",
               "match the official code\".", "", single, ""]

    orders = [("z, z-trans, hilbert, hilbert-trans (the checkpoint's order)", "diag_order_z_z-trans_hilbert_hilbert-trans.json"),
              ("hilbert, hilbert-trans, z, z-trans", "diag_order_hilbert_hilbert-trans_z_z-trans.json"),
              ("hilbert-trans, hilbert, z-trans, z", "diag_order_hilbert-trans_hilbert_z-trans_z.json"),
              ("hilbert, z, hilbert-trans, z-trans", "diag_order_hilbert_z_hilbert-trans_z-trans.json"),
              ("**z, hilbert, z-trans, hilbert-trans** (`FAST`)", "diag_order_z_hilbert_z-trans_hilbert-trans.json")]
    rows = [[name, f"{100 * R(f)['mIoU']:.2f}"] for name, f in orders if R(f)]
    shuffled = [R(f"diag_off_frag_s{i}.json") for i in range(3)]
    if all(shuffled):
        rows.append(["random per stage, as upstream (3 seeds)",
                     " / ".join(f"{100 * r['mIoU']:.2f}" for r in shuffled)])
    if rows:
        md += ["## Curve order at test time", "",
               "Upstream shuffles the four curves at test time as in training, so inference is random. A",
               "deterministic model needs one fixed order, and since block i uses curve i % 4, the 2-block stages",
               "only ever use the first two. Single pass, Area 5:", "",
               table(["Curve order", "mIoU %"], rows), "",
               "Stages that see only one curve family lose 0.2 to 0.6 points. `FAST` alternates the families",
               "(chosen by that rule; the other alternating order scores 0.26 lower, so read differences of a",
               "few tenths here as noise from the choice). The weights don't change, and the order is stored",
               "in the ONNX metadata.", ""]

    par = R("parity_2023.json")
    if par:
        worst = {}
        for r in par:
            w = worst.setdefault(r["backend"], [0.0, 1.0])
            w[0] = max(w[0], r["max_abs_diff"])
            w[1] = min(w[1], r["same_class"])
        md += ["## Parity with the official model", "",
               f"One fragment of each of {len({r['room'] for r in par})} Area 5 rooms, same weights and seed, against the official code "
               "(Dec 2023 release, spconv):", "",
               table(["Backend", "Largest logit difference", "Points with the same class"],
                     [[k, f"{v[0]:.1e}", f"{100 * v[1]:.4f} %"] for k, v in worst.items()]), ""]

    tr = R("train.jsonl")
    newer = {(r["impl"], r["attn"], r["conv"], r["amp"]): r for r in (R("train_rpe.jsonl") or [])}
    if tr:
        tr = [newer.get((r["impl"], r["attn"], r["conv"], r["amp"]), r) for r in tr]
        base = next((r for r in tr if r["impl"] == "official"), None)
        rows = []
        for r in tr:
            name = "official code" if r["impl"] == "official" else f"this repo, {r['attn']} attention + {r['conv']} conv"
            sp = f"{base['ms_per_step'] / r['ms_per_step']:.2f}x" if base else ""
            rows.append([name, r["amp"], f"{r['ms_per_step']:.0f}", sp, f"{r['peak_gpu_gb']:.1f} GB", f"{r['points_per_s'] / 1e6:.2f} M"])
        md += ["## Training step", "",
               f"One optimizer step on a micro-batch of 3 S3DIS training samples (the official recipe's per-GPU batch, "
               f"{tr[0]['mean_points'] / 1e3:.0f}k points on average): forward, CE + Lovász loss, backward, AdamW. Identical "
               "batches for every row.", "",
               table(["Implementation", "Precision", "ms / step", "Speed-up", "Peak GPU memory", "Points / s"], rows), ""]

    fw = (R("forward.jsonl") or []) + (R("forward_onnx.jsonl") or [])
    if fw:
        sizes = sorted({r["points"] for r in fw})
        impls = list(dict.fromkeys(r["impl"] for r in fw))
        get = {(r["impl"], r["points"]): r for r in fw}
        rows = []
        for n in sizes:
            row = [f"{n / 1e3:.0f}k"]
            for i in impls:
                r = get.get((i, n))
                row.append("" if r is None else "out of memory" if "error" in r else f"{r['ms']:.0f} ms / {r['peak_gpu_gb']:.2f} GB")
            rows.append(row)
        md += ["## Inference: time and memory against input size", "",
               "One forward pass, fp32, real S3DIS fragments (one point per 2 cm voxel). PyTorch memory is the",
               "allocator's peak; ONNX Runtime memory is what its process holds after the run beyond what it held",
               "once the model was loaded (its arena keeps the peak). `onnx-fused` runs attention with ONNX Runtime's",
               "fused fp16 kernel; ONNX times exclude the NumPy preparation of the index tables.", "",
               table(["Points"] + impls, rows), ""]

    kn = R("kernels_rpe.jsonl") or R("kernels.jsonl")
    if kn:
        md += ["## The two replaced layers", "",
               f"Stage 0 of a training micro-batch ({kn[0]['points'] / 1e3:.0f}k points, fp16).", "",
               table(["Layer", "Implementation", "Pass", "ms", "Peak GPU memory"],
                     [[r["layer"], r["impl"], r["mode"], f"{r['ms']:.2f}", f"{r['peak_gb']:.2f} GB"] for r in kn
                      # same times as with the cached rulebook to 0.01 ms: not shown as a build measurement
                      if r["impl"] != "spconv incl. rulebook"]), ""]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
