"""Submanifold sparse convolution with plain PyTorch ops (no spconv).

A submanifold convolution computes an output only at the input voxels:

    out[i] = bias + sum_j  W_j @ x[n(i, j)]      over offsets j whose voxel n(i, j) exists

`SubMConv3d` stores its weight exactly as spconv does, (Cout, k, k, k, Cin) with the offsets in
row-major (dx, dy, dz) order, so spconv checkpoints load unchanged.

With backend="torch", two ways to evaluate the sum, chosen per call:
  gather-GEMM  (no grad)  gather the k^3 neighbour rows of a block of voxels into one
               (rows, k^3 * Cin) matrix and multiply once by the flattened weight
  pairs        (grad)     one small GEMM per offset over the voxels that have that neighbour;
               autograd then only does gathers and index_add, never a scatter onto a
               shared zero row (that backward is ~300x slower on GPU)
With backend="triton" (CUDA): an implicit-GEMM kernel, ptv3.kernels.subm_conv.
"""
import itertools

import torch
import torch.nn as nn

KEY_BITS = 16                   # bits per axis in the lookup key: grid coordinates < 65,536
GATHER_BLOCK = 1 << 26          # largest gathered block, in elements (256 MB in fp32)


def kernel_offsets(kernel_size, device=None):
    """(k^3, 3) offsets in spconv's weight order: weight[:, dx, dy, dz, :]."""
    r = kernel_size // 2
    return torch.tensor(list(itertools.product(range(-r, r + 1), repeat=3)), dtype=torch.long, device=device)


@torch.no_grad()
def neighbor_map(grid_coord, batch, kernel_size):
    """(N, k^3) long: entry (i, j) = row of the voxel at grid_coord[i] + offset j, or -1.

    Keys pack (batch, x, y, z) into one int64; the sorted keys are searched once for all
    N * k^3 queries. Duplicate voxels resolve to their lowest row, like a hash table that
    keeps the first insert.
    """
    r = kernel_size // 2
    coords = grid_coord.long() + r                                    # coords + offset >= 0
    assert int(coords.max()) + r < (1 << KEY_BITS), "grid coordinates exceed 16 bits per axis"

    def pack(c, b):
        return (b << (3 * KEY_BITS)) | (c[..., 0] << (2 * KEY_BITS)) | (c[..., 1] << KEY_BITS) | c[..., 2]

    b = batch.long()
    keys = pack(coords, b)
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys[order]
    offsets = kernel_offsets(kernel_size, grid_coord.device)
    nbr = torch.empty(len(keys), len(offsets), dtype=torch.long, device=grid_coord.device)
    rows_per_chunk = max(1, GATHER_BLOCK // (8 * len(offsets)))
    for s in range(0, len(keys), rows_per_chunk):                     # bounded memory for huge inputs
        query = pack(coords[s:s + rows_per_chunk, None, :] + offsets, b[s:s + rows_per_chunk, None])
        pos = torch.searchsorted(sorted_keys, query).clamp_(max=len(keys) - 1)
        nbr[s:s + rows_per_chunk] = torch.where(sorted_keys[pos] == query, order[pos], -1)
    return nbr


@torch.no_grad()
def neighbor_pairs(nbr):
    """Per offset j: (rows, src) = voxels that have a neighbour at offset j, and that neighbour."""
    valid = nbr >= 0
    j, rows = torch.nonzero(valid.t(), as_tuple=True)                 # grouped by offset
    src = nbr[rows, j]
    counts = torch.bincount(j, minlength=nbr.shape[1]).tolist()
    return list(zip(rows.split(counts), src.split(counts)))


def conv_gather_gemm(x, nbr, weight, bias=None):
    """Inference path: out = [x[nbr[:, 0]], ..., x[nbr[:, k^3-1]]] @ W, block by block."""
    n, cin = x.shape
    w = weight.flatten(1).t()                                         # (k^3 * Cin, Cout)
    xz = torch.cat([x, x.new_zeros(1, cin)])                          # row n is zeros: empty neighbour
    idx = torch.where(nbr < 0, n, nbr)
    out = x.new_empty(n, weight.shape[0])
    rows = max(1, GATHER_BLOCK // (nbr.shape[1] * cin))
    for s in range(0, n, rows):
        out[s:s + rows] = xz[idx[s:s + rows]].flatten(1) @ w
    return out if bias is None else out + bias


def conv_pairs(x, pairs, weight, bias=None):
    """Training path: out[rows_j] += x[src_j] @ W_j^T for every offset j (differentiable)."""
    w = weight.flatten(1, 3)                                          # (Cout, k^3, Cin)
    out = x.new_zeros(x.shape[0], w.shape[0])
    for j, (rows, src) in enumerate(pairs):
        if len(rows):
            # rows are unique within one offset, so index_put needs no atomics
            out = out.index_put((rows,), x[src] @ w[:, j].t(), accumulate=True)
    return out if bias is None else out + bias


class SubMConv3d(nn.Module):
    """Drop-in for spconv.SubMConv3d on (features, neighbour map) inputs."""

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, indice_key=None, backend="torch", **_):
        super().__init__()
        assert backend in ("torch", "triton")
        self.backend = backend
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.weight = nn.Parameter(torch.empty(out_channels, kernel_size, kernel_size, kernel_size, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight.view(out_channels, -1), a=5 ** 0.5)

    def tables(self, point):
        """Neighbour map (and pairs, when training) of this layer's stage, cached on the Point."""
        cache = point.setdefault("conv_cache", {})
        key = (self.indice_key, self.kernel_size)
        if key not in cache:
            cache[key] = {"nbr": neighbor_map(point.grid_coord, point.batch, self.kernel_size)}
        return cache[key]

    def forward(self, point, feat):
        t = self.tables(point)
        if self.backend == "triton":
            from .kernels.subm_conv import subm_conv_triton
            return subm_conv_triton(feat, t["nbr"], self.weight, self.bias)
        if torch.is_grad_enabled() and (feat.requires_grad or self.weight.requires_grad):
            if "pairs" not in t:
                t["pairs"] = neighbor_pairs(t["nbr"])
            return conv_pairs(feat, t["pairs"], self.weight, self.bias)
        weight = self.weight.to(feat.dtype)
        bias = None if self.bias is None else self.bias.to(feat.dtype)
        return conv_gather_gemm(feat, t["nbr"], weight, bias)
