"""Space-filling-curve codes (z-order and Hilbert) on integer tensors.

Same codes as Pointcept's `serialization.encode`, computed with integer bit tricks instead of
lookup tables (z-order) and per-bit boolean tensors (Hilbert). The Hilbert encoder is Skilling's
algorithm (2004), the one Pointcept's bit-array version implements, applied to whole integers:
it allocates (N,) tensors instead of (N, 3, depth) ones. `tests/test_serialization.py` checks
both encoders against the official ones bit for bit.
"""
import torch

ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")


def _spread3(v):
    """Move bit i of a 16-bit value to bit 3i (so three values can be interleaved)."""
    v = v & 0xFFFF
    v = (v | (v << 16)) & 0x0000FF0000FF
    v = (v | (v << 8)) & 0x00F00F00F00F
    v = (v | (v << 4)) & 0x0C30C30C30C3
    v = (v | (v << 2)) & 0x249249249249
    return v


def z_order(grid):
    """(N, 3) int64 -> (N,) int64, x in the most significant bit of every triple."""
    return (_spread3(grid[:, 0]) << 2) | (_spread3(grid[:, 1]) << 1) | _spread3(grid[:, 2])


def hilbert(grid, depth):
    """(N, 3) int64 grid coordinates < 2**depth -> (N,) int64 Hilbert index."""
    x = [grid[:, 0].clone(), grid[:, 1].clone(), grid[:, 2].clone()]
    q = 1 << (depth - 1)
    while q > 1:                                    # from the top bit down to bit 1
        p = q - 1                                   # mask of the bits below q
        for i in range(3):
            high = (x[i] & q) != 0
            x[0] = torch.where(high, x[0] ^ p, x[0])            # bit set: invert the low bits of x0
            t = torch.where(high, torch.zeros_like(x[0]), (x[0] ^ x[i]) & p)
            x[0] = x[0] ^ t                                     # bit clear: swap low bits of x0, xi
            x[i] = x[i] ^ t
        q >>= 1
    x[1] = x[1] ^ x[0]                              # Gray decode across axes ...
    x[2] = x[2] ^ x[1]
    t = torch.zeros_like(x[0])
    q = 1 << (depth - 1)
    while q > 1:                                    # ... and across bit levels
        t = torch.where((x[2] & q) != 0, t ^ (q - 1), t)
        q >>= 1
    x = [xi ^ t for xi in x]
    return z_order(torch.stack(x, dim=1))          # interleave, x0 most significant


@torch.no_grad()
def encode(grid_coord, batch=None, depth=16, order="z"):
    """Serialization code of every point; the batch index goes above the 3 * depth code bits."""
    assert order in ORDERS, order
    assert 1 <= depth <= 16, depth
    grid = grid_coord.long()
    if order.endswith("-trans"):
        grid = grid[:, [1, 0, 2]]
    code = z_order(grid) if order == "z" or order == "z-trans" else hilbert(grid, depth)
    if batch is not None:
        code = (batch.long() << (3 * depth)) | code
    return code


def encode_per_sample(grid_coord, batch, sample_depth, depth, order="z"):
    """encode() with each sample's Hilbert curve at its own depth (sample_depth[b]); the batch index
    goes above 3 * depth bits, depth >= every sample depth. z-order doesn't depend on the depth."""
    if order in ("z", "z-trans"):
        return encode(grid_coord, batch, depth, order)
    code = torch.empty(len(grid_coord), dtype=torch.long, device=grid_coord.device)
    per_point = torch.tensor(sample_depth, device=grid_coord.device)[batch.long()]
    for d in set(sample_depth):
        rows = per_point == d
        code[rows] = encode(grid_coord[rows], None, d, order)
    return (batch.long() << (3 * depth)) | code
