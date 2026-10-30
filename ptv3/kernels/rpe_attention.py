"""Patch attention with PTv3's relative position encoding, fused in Triton (forward and backward).

Upstream computes RPE attention as:
    rel   = grid[i] - grid[j]                      (P, K, K, 3) int
    bias  = table[clamp(rel) + offsets].sum(axis)  gathers a (P, K, K, 3, H) tensor, sums to (P, H, K, K)
    attn  = softmax(q k^T * scale + bias) v        materializes (P, H, K, K) scores and probabilities
and autograd keeps those tensors for the backward pass. This kernel loads the voxel coordinates of
a patch, computes the bias of every (query, key) pair in registers, and never writes a (K, K) tensor:
memory is O(P * K * C) like flash attention.

One program handles one (patch, head) pair; a patch has at most K = 128 points, so the whole key
set fits in one tile and the softmax needs no online rescaling. The backward pass recomputes the
probabilities from the saved log-sum-exp. The RPE table gradient is reduced per bin inside each
program (one masked sum per bin), then added to the table with one atomic per bin and program.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _bias(s, grid_ptr, offs_m, offs_n, K, table_ptr, h, H: tl.constexpr, BND: tl.constexpr, NB: tl.constexpr):
    for a in tl.static_range(3):
        gi = tl.load(grid_ptr + offs_m * 3 + a, mask=offs_m < K, other=0)
        gj = tl.load(grid_ptr + offs_n * 3 + a, mask=offs_n < K, other=0)
        rel = tl.minimum(tl.maximum(gi[:, None] - gj[None, :], -BND), BND) + BND
        s += tl.load(table_ptr + (rel + a * NB) * H + h)
    return s


@triton.jit(do_not_specialize=["K", "scale"])
def _fwd_kernel(qkv_ptr, grid_ptr, table_ptr, len_ptr, out_ptr, lse_ptr, scale, K,
                H: tl.constexpr, D: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr,
                RPE: tl.constexpr, BND: tl.constexpr, NB: tl.constexpr, MASKED: tl.constexpr,
                PRECISION: tl.constexpr):
    p = tl.program_id(0)
    h = tl.program_id(1)
    offs_m = tl.program_id(2) * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BK)
    offs_d = tl.arange(0, D)
    row = 3 * H * D                                              # qkv is (P, K, 3, H, D)
    base = qkv_ptr + p.to(tl.int64) * K * row + h * D
    q = tl.load(base + offs_m[:, None] * row + offs_d[None, :], mask=offs_m[:, None] < K, other=0.0)
    k = tl.load(base + H * D + offs_n[:, None] * row + offs_d[None, :], mask=offs_n[:, None] < K, other=0.0)
    v = tl.load(base + 2 * H * D + offs_n[:, None] * row + offs_d[None, :], mask=offs_n[:, None] < K, other=0.0)

    s = tl.dot(q, tl.trans(k), input_precision=PRECISION) * scale          # (BM, BK) fp32
    if RPE:
        s = _bias(s, grid_ptr + p.to(tl.int64) * K * 3, offs_m, offs_n, K, table_ptr, h, H, BND, NB)
    n_valid = tl.load(len_ptr + p) if MASKED else K
    s = tl.where((offs_n < n_valid)[None, :], s, float("-inf"))
    m = tl.max(s, axis=1)
    e = tl.exp(s - m[:, None])
    l = tl.sum(e, axis=1)
    o = tl.dot(e.to(v.dtype), v, input_precision=PRECISION) / l[:, None]

    out = out_ptr + p.to(tl.int64) * K * H * D + h * D
    tl.store(out + offs_m[:, None] * H * D + offs_d[None, :], o.to(out_ptr.dtype.element_ty), mask=offs_m[:, None] < K)
    tl.store(lse_ptr + (p.to(tl.int64) * H + h) * K + offs_m, m + tl.log(l), mask=offs_m < K)


@triton.jit(do_not_specialize=["K", "scale"])
def _bwd_kernel(qkv_ptr, grid_ptr, table_ptr, len_ptr, dout_ptr, lse_ptr, delta_ptr, dqkv_ptr, dtable_ptr, scale, K,
                H: tl.constexpr, D: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr,
                RPE: tl.constexpr, BND: tl.constexpr, NB: tl.constexpr, NBP: tl.constexpr, MASKED: tl.constexpr,
                PRECISION: tl.constexpr):
    p = tl.program_id(0)
    h = tl.program_id(1)
    offs_n = tl.arange(0, BK)
    offs_d = tl.arange(0, D)
    row = 3 * H * D
    base = qkv_ptr + p.to(tl.int64) * K * row + h * D
    dbase = dqkv_ptr + p.to(tl.int64) * K * row + h * D
    obase = dout_ptr + p.to(tl.int64) * K * H * D + h * D
    grid = grid_ptr + p.to(tl.int64) * K * 3
    k = tl.load(base + H * D + offs_n[:, None] * row + offs_d[None, :], mask=offs_n[:, None] < K, other=0.0)
    v = tl.load(base + 2 * H * D + offs_n[:, None] * row + offs_d[None, :], mask=offs_n[:, None] < K, other=0.0)
    n_valid = tl.load(len_ptr + p) if MASKED else K
    key_ok = (offs_n < n_valid)[None, :]
    dk = tl.zeros([BK, D], dtype=tl.float32)
    dv = tl.zeros([BK, D], dtype=tl.float32)
    bins = tl.arange(0, NBP)
    g0 = tl.zeros([NBP], dtype=tl.float32)
    g1 = tl.zeros([NBP], dtype=tl.float32)
    g2 = tl.zeros([NBP], dtype=tl.float32)

    for start in tl.static_range(0, BK, BM):
        offs_m = start + tl.arange(0, BM)
        rows = offs_m[:, None] < K
        q = tl.load(base + offs_m[:, None] * row + offs_d[None, :], mask=rows, other=0.0)
        do = tl.load(obase + offs_m[:, None] * H * D + offs_d[None, :], mask=rows, other=0.0)
        lse = tl.load(lse_ptr + (p.to(tl.int64) * H + h) * K + offs_m, mask=offs_m < K, other=0.0)
        delta = tl.load(delta_ptr + (p.to(tl.int64) * H + h) * K + offs_m, mask=offs_m < K, other=0.0)
        s = tl.dot(q, tl.trans(k), input_precision=PRECISION) * scale
        if RPE:
            s = _bias(s, grid, offs_m, offs_n, K, table_ptr, h, H, BND, NB)
        prob = tl.where(key_ok & rows, tl.exp(s - lse[:, None]), 0.0)          # (BM, BK)
        dv += tl.dot(tl.trans(prob).to(do.dtype), do, input_precision=PRECISION)
        dp = tl.dot(do, tl.trans(v), input_precision=PRECISION)
        ds = prob * (dp - delta[:, None])                                      # d loss / d score
        dq = tl.dot(ds.to(k.dtype), k, input_precision=PRECISION) * scale
        tl.store(dbase + offs_m[:, None] * row + offs_d[None, :], dq.to(dqkv_ptr.dtype.element_ty), mask=rows)
        dk += tl.dot(tl.trans(ds).to(q.dtype), q, input_precision=PRECISION) * scale
        if RPE:   # d table[axis, bin, h] = sum of ds over the pairs that fall in that bin
            for a in tl.static_range(3):
                gi = tl.load(grid + offs_m * 3 + a, mask=offs_m < K, other=0)
                gj = tl.load(grid + offs_n * 3 + a, mask=offs_n < K, other=0)
                rel = tl.minimum(tl.maximum(gi[:, None] - gj[None, :], -BND), BND) + BND
                acc = tl.zeros([NBP], dtype=tl.float32)
                for b in tl.static_range(NB):
                    acc += tl.where(bins == b, tl.sum(tl.where(rel == b, ds, 0.0)), 0.0)
                if a == 0:
                    g0 += acc
                elif a == 1:
                    g1 += acc
                else:
                    g2 += acc

    tl.store(dbase + H * D + offs_n[:, None] * row + offs_d[None, :], dk.to(dqkv_ptr.dtype.element_ty), mask=offs_n[:, None] < K)
    tl.store(dbase + 2 * H * D + offs_n[:, None] * row + offs_d[None, :], dv.to(dqkv_ptr.dtype.element_ty), mask=offs_n[:, None] < K)
    if RPE:
        tl.atomic_add(dtable_ptr + bins * H + h, g0, mask=bins < NB)
        tl.atomic_add(dtable_ptr + (NB + bins) * H + h, g1, mask=bins < NB)
        tl.atomic_add(dtable_ptr + (2 * NB + bins) * H + h, g2, mask=bins < NB)


def _launch_cfg(K, qkv):
    BK = max(16, triton.next_power_of_2(K))
    BM = min(BK, 64)
    precision = "ieee" if qkv.dtype == torch.float32 else "tf32"
    return BK, BM, precision


class _RPEAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, grid, table, lengths, scale, bnd, nb):
        P, K, _, H, D = qkv.shape
        BK, BM, precision = _launch_cfg(K, qkv)
        out = torch.empty(P, K, H, D, device=qkv.device, dtype=qkv.dtype)
        lse = torch.empty(P, H, K, device=qkv.device, dtype=torch.float32)
        rpe, masked = table is not None, lengths is not None
        dummy = qkv.new_empty(1)
        _fwd_kernel[(P, H, triton.cdiv(K, BM))](
            qkv, grid if rpe else dummy, table if rpe else dummy, lengths if masked else dummy, out, lse,
            scale, K, H=H, D=D, BK=BK, BM=BM, RPE=rpe, BND=bnd, NB=nb, MASKED=masked, PRECISION=precision)
        ctx.save_for_backward(qkv, grid, table, lengths, out, lse)
        ctx.args = (scale, bnd, nb, rpe, masked)
        return out

    @staticmethod
    def backward(ctx, dout):
        qkv, grid, table, lengths, out, lse = ctx.saved_tensors
        scale, bnd, nb, rpe, masked = ctx.args
        P, K, _, H, D = qkv.shape
        BK, BM, precision = _launch_cfg(K, qkv)
        dout = dout.contiguous()
        delta = (dout.float() * out.float()).sum(-1).transpose(1, 2).contiguous()     # (P, H, K)
        dqkv = torch.empty_like(qkv)
        dtable = torch.zeros(3 * nb, H, device=qkv.device, dtype=torch.float32) if rpe else None
        dummy = qkv.new_empty(1)
        _bwd_kernel[(P, H)](
            qkv, grid if rpe else dummy, table if rpe else dummy, lengths if masked else dummy, dout, lse, delta,
            dqkv, dtable if rpe else dummy, scale, K, H=H, D=D, BK=BK, BM=BM, RPE=rpe, BND=bnd, NB=nb,
            NBP=max(16, triton.next_power_of_2(nb)), MASKED=masked, PRECISION=precision)
        return dqkv, None, (dtable.to(table.dtype) if rpe else None), None, None, None, None


def patch_attention_rpe(qkv_dense, grid, rpe, key_mask, num_heads, scale):
    """qkv_dense (P, K, 3C) gathered patch slots, grid (P, K, 3) voxel coords or None,
    rpe: ptv3.attention.RPE or None, key_mask (P, K) or None -> (P * K, C)."""
    P, K, C3 = qkv_dense.shape
    D = C3 // 3 // num_heads
    qkv = qkv_dense.reshape(P, K, 3, num_heads, D).contiguous()
    lengths = key_mask.sum(1, dtype=torch.int32) if key_mask is not None else None
    if rpe is not None:
        out = _RPEAttention.apply(qkv, grid.to(torch.int32).contiguous(), rpe.rpe_table, lengths, scale,
                                  rpe.pos_bnd, rpe.rpe_num)
    else:
        out = _RPEAttention.apply(qkv, None, None, lengths, scale, 0, 1)
    return out.reshape(P * K, -1)
