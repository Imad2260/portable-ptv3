"""Submanifold sparse convolution as an implicit GEMM in Triton (forward and backward).

    out[i] = sum_j x[nbr[i, j]] @ W_j            W_j = weight[:, j, :]^T, (Cin, Cout)

The forward kernel computes a (rows x Cout) tile per program: for every offset j it gathers the
neighbour rows of x straight from memory and multiplies them by W_j, so the (N, k^3, Cin) gathered
matrix of the plain PyTorch path is never written. Offsets with no neighbour in a whole row block
are skipped (on sparse inputs most are).

Backward:
  dX[nbr[i, j]] += dY[i] @ W_j^T     atomic adds: correct even when two rows share a voxel (mixed
                                     training samples), where a voxel's lookups all resolve to one row
  dW_j = sum_i x[nbr[i, j]]^T dY[i]  rows split across programs, partial sums added atomically
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _fwd_kernel(x_ptr, nbr_ptr, w_ptr, out_ptr, N, KV,
                CIN: tl.constexpr, COUT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BC: tl.constexpr,
                PRECISION: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    row_ok = rows < N
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for j in range(KV):
        idx = tl.load(nbr_ptr + rows.to(tl.int64) * KV + j, mask=row_ok, other=-1)
        valid = idx >= 0
        if tl.sum(valid.to(tl.int32)) > 0:
            for c0 in tl.static_range(0, CIN, BC):
                cin = c0 + tl.arange(0, BC)
                xg = tl.load(x_ptr + idx.to(tl.int64)[:, None] * CIN + cin[None, :],
                             mask=valid[:, None] & (cin[None, :] < CIN), other=0.0)
                w = tl.load(w_ptr + (j * CIN + cin[:, None]) * COUT + cols[None, :],
                            mask=(cin[:, None] < CIN) & (cols[None, :] < COUT), other=0.0)
                acc += tl.dot(xg, w, input_precision=PRECISION)
    tl.store(out_ptr + rows.to(tl.int64)[:, None] * COUT + cols[None, :], acc.to(out_ptr.dtype.element_ty),
             mask=row_ok[:, None] & (cols[None, :] < COUT))


@triton.jit(do_not_specialize=["N"])
def _bwd_x_kernel(dy_ptr, nbr_ptr, w_ptr, dx_ptr, N, KV,
                  CIN: tl.constexpr, COUT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BC: tl.constexpr,
                  PRECISION: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cin = tl.program_id(1) * BC + tl.arange(0, BC)
    row_ok = rows < N
    for j in range(KV):
        idx = tl.load(nbr_ptr + rows.to(tl.int64) * KV + j, mask=row_ok, other=-1)
        valid = idx >= 0
        if tl.sum(valid.to(tl.int32)) > 0:
            acc = tl.zeros([BM, BC], dtype=tl.float32)
            for n0 in tl.static_range(0, COUT, BN):
                cols = n0 + tl.arange(0, BN)
                dy = tl.load(dy_ptr + rows.to(tl.int64)[:, None] * COUT + cols[None, :],
                             mask=row_ok[:, None] & (cols[None, :] < COUT), other=0.0)
                wt = tl.load(w_ptr + (j * CIN + cin[None, :]) * COUT + cols[:, None],          # W_j^T: (BN, BC)
                             mask=(cin[None, :] < CIN) & (cols[:, None] < COUT), other=0.0)
                acc += tl.dot(dy, wt, input_precision=PRECISION)
            tl.atomic_add(dx_ptr + idx.to(tl.int64)[:, None] * CIN + cin[None, :], acc,
                          mask=valid[:, None] & (cin[None, :] < CIN))


@triton.jit(do_not_specialize=["N", "ROWS_PER_PROG", "SPLIT"])
def _bwd_w_kernel(x_ptr, dy_ptr, nbr_ptr, dw_ptr, N, KV, ROWS_PER_PROG, SPLIT,
                  CIN: tl.constexpr, COUT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BC: tl.constexpr,
                  PRECISION: tl.constexpr):
    j = tl.program_id(0) // SPLIT                       # axis 0 = (offset, row range)
    start = (tl.program_id(0) % SPLIT) * ROWS_PER_PROG
    cin = tl.program_id(1) * BC + tl.arange(0, BC)
    cols = tl.program_id(2) * BN + tl.arange(0, BN)
    acc = tl.zeros([BC, BN], dtype=tl.float32)
    for r0 in range(start, start + ROWS_PER_PROG, BM):
        rows = r0 + tl.arange(0, BM)
        row_ok = (rows < N) & (rows < start + ROWS_PER_PROG)
        idx = tl.load(nbr_ptr + rows.to(tl.int64) * KV + j, mask=row_ok, other=-1)
        valid = idx >= 0
        xg = tl.load(x_ptr + idx.to(tl.int64)[:, None] * CIN + cin[None, :],
                     mask=valid[:, None] & (cin[None, :] < CIN), other=0.0)
        dy = tl.load(dy_ptr + rows.to(tl.int64)[:, None] * COUT + cols[None, :],
                     mask=valid[:, None] & (cols[None, :] < COUT), other=0.0)
        acc += tl.dot(tl.trans(xg), dy, input_precision=PRECISION)
    tl.atomic_add(dw_ptr + (j * CIN + cin[:, None]) * COUT + cols[None, :], acc,
                  mask=(cin[:, None] < CIN) & (cols[None, :] < COUT))


def _blocks(cin, cout, dtype):
    bc = min(64, max(16, triton.next_power_of_2(cin)))
    bn = min(64, max(16, triton.next_power_of_2(cout)))
    return 64, bn, bc, ("ieee" if dtype == torch.float32 else "tf32")


class _SubMConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, nbr, weight):
        """x (N, Cin), nbr (N, k^3) int64 with -1 for empty, weight (Cout, k, k, k, Cin) -> (N, Cout)."""
        n, cin = x.shape
        cout, kv = weight.shape[0], nbr.shape[1]
        w = weight.reshape(cout, kv, cin).permute(1, 2, 0).contiguous()        # (k^3, Cin, Cout)
        out = torch.empty(n, cout, device=x.device, dtype=x.dtype)
        BM, BN, BC, prec = _blocks(cin, cout, x.dtype)
        _fwd_kernel[(triton.cdiv(n, BM), triton.cdiv(cout, BN))](
            x, nbr, w, out, n, kv, CIN=cin, COUT=cout, BM=BM, BN=BN, BC=BC, PRECISION=prec)
        ctx.save_for_backward(x, nbr, w)
        ctx.weight_shape, ctx.weight_dtype = weight.shape, weight.dtype
        return out

    @staticmethod
    def backward(ctx, dy):
        x, nbr, w = ctx.saved_tensors
        n, cin = x.shape
        kv, _, cout = w.shape
        dy = dy.contiguous().to(x.dtype)
        BM, BN, BC, prec = _blocks(cin, cout, x.dtype)
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dx32 = torch.zeros(n, cin, device=x.device, dtype=torch.float32)
            _bwd_x_kernel[(triton.cdiv(n, BM), triton.cdiv(cin, BC))](
                dy, nbr, w, dx32, n, kv, CIN=cin, COUT=cout, BM=BM, BN=BN, BC=BC, PRECISION=prec)
            dx = dx32.to(x.dtype)
        if ctx.needs_input_grad[2]:
            dw32 = torch.zeros(kv, cin, cout, device=x.device, dtype=torch.float32)
            rows = triton.cdiv(triton.cdiv(n, max(1, min(256, n // 2048))), BM) * BM
            split = triton.cdiv(n, rows)
            _bwd_w_kernel[(kv * split, triton.cdiv(cin, BC), triton.cdiv(cout, BN))](
                x, dy, nbr, dw32, n, kv, rows, split, CIN=cin, COUT=cout, BM=BM, BN=BN, BC=BC, PRECISION=prec)
            dw = dw32.permute(2, 0, 1).reshape(ctx.weight_shape).to(ctx.weight_dtype)
        return dx, None, dw


def subm_conv_triton(x, nbr, weight, bias=None):
    if torch.is_autocast_enabled():
        dtype = torch.get_autocast_dtype("cuda") if x.is_cuda else torch.get_autocast_dtype("cpu")
        x, weight = x.to(dtype), weight.to(dtype)
        bias = None if bias is None else bias.to(dtype)
    with torch.autocast(device_type="cuda", enabled=False):
        out = _SubMConv.apply(x.contiguous(), nbr.contiguous(), weight)
    return out if bias is None else out + bias
