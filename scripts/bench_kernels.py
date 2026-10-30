"""Micro-benchmarks of the two layers that replace CUDA extensions, at the sizes of a real S3DIS batch.

    python scripts/bench_kernels.py --data-root data/s3dis --official-repo third_party/PointTransformerV3

Attention: one RPE attention layer of stage 0 (C = 32, 2 heads, K = 128) on a training micro-batch of
3 S3DIS samples (about 400k points, the shapes upstream trains on): upstream's math path (RPE gather +
matmul + softmax), SDPA with the bias, and the fused Triton kernel; forward, and forward + backward.
Convolution: one 3^3 CPE convolution (C = 32) on the same voxels: spconv, the PyTorch paths, Triton.
Index structures (spconv's rulebook, the neighbour map here) are built once and reused, as both
implementations do across the blocks of a stage; their build time is reported on separate rows.
Times are medians of 20 runs after 5 warm-ups (CUDA events); memory is the peak above the inputs.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ptv3.attention import RPE, SerializedAttention  # noqa: E402
from ptv3.model import Point  # noqa: E402
from ptv3.serialization import encode  # noqa: E402
from ptv3.sparse_conv import SubMConv3d, neighbor_map  # noqa: E402
from ptv3.train import collate  # noqa: E402
from ptv3.data import s3dis  # noqa: E402


def measure(fn, backward=False, reps=20, warmup=5):
    for _ in range(warmup):
        out = fn()
        if backward:
            out.float().sum().backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        out = fn()
        if backward:
            out.float().sum().backward()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return float(np.median(times)), (torch.cuda.max_memory_allocated() - base) / 2**30


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--official-repo", default=None, help="for spconv and upstream's attention module")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    np.random.seed(0)
    data = s3dis.S3DIS(args.data_root, loop=1)
    batch = collate([data[i] for i in range(args.samples)], mix_prob=0.0)
    grid = batch["grid_coord"].cuda()
    offset = batch["offset"].cuda()
    bidx = torch.repeat_interleave(torch.arange(len(offset), device="cuda"), torch.diff(offset, prepend=offset.new_zeros(1)))
    n = len(grid)
    depth = int(grid.max()).bit_length()
    code = encode(grid, bidx, depth, "hilbert")[None]
    order = torch.argsort(code)
    inverse = torch.empty_like(order).scatter_(1, order, torch.arange(n, device="cuda")[None])
    results = []

    def report(layer, impl, mode, ms, gb):
        rec = {"layer": layer, "impl": impl, "mode": mode, "points": n, "dtype": args.dtype, "ms": round(ms, 3), "peak_gb": round(gb, 3)}
        results.append(rec)
        print(json.dumps(rec), flush=True)

    # ---------------- attention
    C, H, K = 32, 2, 128
    torch.manual_seed(0)
    feat = torch.randn(n, C, device="cuda", requires_grad=True)
    impls = {}
    for backend in ("math", "sdpa", "triton"):
        m = SerializedAttention(C, H, K, enable_rpe=True, attn_backend=backend, patch_mode="min").cuda()
        impls[backend] = m
    state = impls["math"].state_dict()
    for m in impls.values():
        m.load_state_dict(state)
    if args.official_repo:
        sys.path.insert(0, os.path.dirname(os.path.abspath(args.official_repo)))
        om = __import__(os.path.basename(os.path.abspath(args.official_repo)) + ".model", fromlist=["SerializedAttention"])
        up = om.SerializedAttention(C, H, K, enable_rpe=True, enable_flash=False, upcast_attention=True, upcast_softmax=True).cuda()
        up.load_state_dict(state)
        impls = {"upstream": up, **impls}

    def attn_fn(m, official):
        def f():
            if official:
                p = om.Point(feat=feat, offset=offset, grid_coord=grid, serialized_order=order, serialized_inverse=inverse, batch=bidx)
            else:
                p = Point(feat=feat, offset=offset, grid_coord=grid, serialized_order=order, serialized_inverse=inverse)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                return m(p).feat
        return f

    for name, m in impls.items():
        for mode, bw in (("forward", False), ("forward+backward", True)):
            try:
                if not bw:
                    with torch.no_grad():
                        ms, gb = measure(attn_fn(m, name == "upstream"))
                else:
                    ms, gb = measure(attn_fn(m, name == "upstream"), backward=True)
                report("rpe-attention stage 0", name, mode, ms, gb)
            except torch.cuda.OutOfMemoryError:
                report("rpe-attention stage 0", name, mode, float("nan"), float("nan"))
                torch.cuda.empty_cache()

    # ---------------- convolution
    conv = SubMConv3d(C, C, 3).cuda()
    nbr = neighbor_map(grid, bidx, 3)
    point = Point(grid_coord=grid, batch=bidx, offset=offset)

    def ours(backend):
        def f():
            conv.backend = backend
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                return conv(point, feat)
        return f

    convs = {"torch (gather-GEMM / pairs)": ours("torch"), "triton": ours("triton")}
    if args.official_repo:
        import spconv.pytorch as spconv
        sc = spconv.SubMConv3d(C, C, 3, bias=True, indice_key="b").cuda()
        sc.weight.data.copy_(conv.weight.data)
        sc.bias.data.copy_(conv.bias.data)
        idx = torch.cat([bidx[:, None].int(), grid.int()], 1).contiguous()
        shape = (grid.max(0).values + 96).tolist()

        base = spconv.SparseConvTensor(feat, idx, shape, len(offset))
        with torch.no_grad():
            sc(base)                                   # builds the rulebook once, cached in base.indice_dict

        def spc():                                     # rulebook reused, like ours reuses its neighbour map
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                return sc(base.replace_feature(feat)).features

        def spc_fresh():                               # including the rulebook build
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                return sc(spconv.SparseConvTensor(feat, idx, shape, len(offset))).features
        convs = {"spconv": spc, "spconv incl. rulebook": spc_fresh, **convs}
    for name, fn in convs.items():
        for mode, bw in (("forward", False), ("forward+backward", True)):
            if not bw:
                with torch.no_grad():
                    ms, gb = measure(fn)
            else:
                ms, gb = measure(fn, backward=True)
            report("subm-conv 3^3 C=32", name, mode, ms, gb)
    with torch.no_grad():
        ms, gb = measure(lambda: neighbor_map(grid, bidx, 3))
    report("neighbour map 3^3", "sort + searchsorted", "build", ms, gb)
    if args.out:
        with open(args.out, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
