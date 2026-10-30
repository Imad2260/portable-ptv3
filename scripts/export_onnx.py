"""Export a PTv3 checkpoint to ONNX (standard operators, opset 18, every index dimension dynamic).

    python scripts/export_onnx.py --checkpoint weights/s3dis/model_best.pth --out ptv3_s3dis.onnx

The ONNX model takes the tables of ptv3.export.prep (NumPy) and returns logits. The model settings
the preprocessing needs travel in the file's metadata ("ptv3_config"), so the runtime needs only
the .onnx file.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.export import Dim  # noqa: E402

from ptv3.checkpoint import load  # noqa: E402
from ptv3.export.graph import ExportGraph  # noqa: E402
from ptv3.export.prep import collate, prepare  # noqa: E402
from ptv3.presets import FAST, PRESETS  # noqa: E402

FIXED_LAST_AXIS = ("feat0", "_nbr", "_children", "_grid")      # their last axis is a constant (C, k^3, 8, 3)


def multi_head_attention_translation():
    """Map ptv3::mha to ONNX Runtime's com.microsoft.MultiHeadAttention (query, key, value, bias,
    key_padding_mask, attention_bias). The onnx package has no schema for this contrib op, so declare
    the inputs used here before onnxscript can emit it."""
    import onnx
    import onnxscript
    from onnx.defs import OpSchema

    P, A = OpSchema.FormalParameter, OpSchema.Attribute
    opt = OpSchema.FormalParameterOption.Optional
    onnx.defs.register_schema(OpSchema(
        "MultiHeadAttention", "com.microsoft", 1,
        inputs=[P("query", "T"), P("key", "T"), P("value", "T"), P("bias", "T", param_option=opt),
                P("key_padding_mask", "M", param_option=opt), P("attention_bias", "T", param_option=opt)],
        outputs=[P("output", "T")],
        type_constraints=[("T", ["tensor(float)", "tensor(float16)"], ""), ("M", ["tensor(int32)"], "")],
        attributes=[A("num_heads", OpSchema.AttrType.INT, ""), A("scale", OpSchema.AttrType.FLOAT, "")],
    ))
    msft = onnxscript.values.Opset("com.microsoft", 1)

    def mha(q, k, v, bias, num_heads: int, scale: float):
        return msft.MultiHeadAttention(q, k, v, None, None, bias, num_heads=num_heads, scale=scale)
    return {torch.ops.ptv3.mha.default: mha}


def example_tables(cfg, in_channels):
    """Two synthetic scenes of different sizes: realistic shapes for tracing, no dataset needed."""
    rng = np.random.default_rng(0)
    samples = []
    for n in (60000, 20000):
        pts = rng.uniform(0, 1, (n, 3)) * [6.0, 4.0, 3.0]
        pts[: n // 2, 2] = 0                                         # a floor, so voxels have neighbours
        grid = np.unique(np.floor(pts / 0.02).astype(np.int64), axis=0)
        samples.append(prepare(grid, rng.normal(size=(len(grid), in_channels)).astype(np.float32), cfg))
    return collate(samples, cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--preset", default="s3dis-rpe", choices=sorted(PRESETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--fused-attention", action="store_true",
                    help="attention as ONNX Runtime's fused MultiHeadAttention (fp16): no K x K scores in memory")
    args = ap.parse_args()

    options = {**FAST, "attn_backend": "sdpa"}
    model = load(args.preset, args.checkpoint, **options)
    cfg = {**PRESETS[args.preset], **options}
    tables = example_tables(cfg, cfg["in_channels"])
    names = list(tables)
    graph = ExportGraph(model, names, fused_attention=args.fused_attention).eval()
    inputs = tuple(torch.from_numpy(tables[n]) for n in names)
    dynamic = (tuple({d: Dim.DYNAMIC for d in range(tables[n].ndim - (1 if n.endswith(FIXED_LAST_AXIS) else 0))}
                     for n in names),)
    with torch.no_grad():
        program = torch.onnx.export(graph, inputs, dynamo=True, opset_version=18, input_names=names,
                                    output_names=["logits"], dynamic_shapes=dynamic,
                                    custom_translation_table=multi_head_attention_translation() if args.fused_attention else None)
    meta = {k: cfg[k] for k in ("order", "enc_depths", "dec_depths", "enc_patch_size", "dec_patch_size",
                                "patch_mode", "serialization_depth", "in_channels", "num_classes")}
    program.model.metadata_props["ptv3_config"] = json.dumps(meta)
    program.save(args.out)
    print(f"{args.out}: {len(names)} inputs, {os.path.getsize(args.out) / 1e6:.0f} MB, fused attention: {args.fused_attention}")


if __name__ == "__main__":
    main()
