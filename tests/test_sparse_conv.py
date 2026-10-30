import os

import pytest
import torch

from conftest import make_batch
from ptv3.model import offset2batch
from ptv3.sparse_conv import SubMConv3d, conv_gather_gemm, conv_pairs, kernel_offsets, neighbor_map, neighbor_pairs


def reference_conv(feat, grid, batch, weight, bias):
    """out[i] = bias + sum over existing neighbours j of W_j x[j], with a Python dict of voxels."""
    k = weight.shape[1]
    lookup = {(int(b), *map(int, g)): i for i, (b, g) in enumerate(zip(batch, grid))}
    w = weight.flatten(1, 3)                                    # (Cout, k^3, Cin)
    out = bias.repeat(len(feat), 1).double()
    for i in range(len(feat)):
        for j, off in enumerate(kernel_offsets(k).tolist()):
            n = lookup.get((int(batch[i]), *(int(g) + o for g, o in zip(grid[i], off))))
            if n is not None:
                out[i] += w[:, j].double() @ feat[n].double()
    return out


def test_neighbor_map_matches_dict_lookup():
    b = make_batch([600, 400])
    batch = offset2batch(b["offset"])
    nbr = neighbor_map(b["grid_coord"], batch, 3)
    lookup = {(int(bb), *map(int, g)): i for i, (bb, g) in enumerate(zip(batch, b["grid_coord"]))}
    for i in range(0, len(batch), 7):
        for j, off in enumerate(kernel_offsets(3).tolist()):
            want = lookup.get((int(batch[i]), *(int(g) + o for g, o in zip(b["grid_coord"][i], off))), -1)
            assert nbr[i, j] == want
    assert (nbr[:, 13] == torch.arange(len(batch))).all()       # the centre tap is the voxel itself
    assert (nbr >= 0).float().mean() > 0.2                      # surfaces: the map is not trivially empty


def test_conv_paths_match_reference():
    torch.manual_seed(0)
    b = make_batch([300, 200], in_channels=5)
    batch = offset2batch(b["offset"])
    for k in (3, 5):
        conv = SubMConv3d(5, 7, k, bias=True)
        conv.bias.data.normal_()
        nbr = neighbor_map(b["grid_coord"], batch, k)
        ref = reference_conv(b["feat"], b["grid_coord"], batch, conv.weight.data, conv.bias.data)
        with torch.no_grad():
            a = conv_gather_gemm(b["feat"], nbr, conv.weight, conv.bias)
            p = conv_pairs(b["feat"], neighbor_pairs(nbr), conv.weight, conv.bias)
        assert torch.allclose(a.double(), ref, atol=1e-4)
        assert torch.allclose(p.double(), ref, atol=1e-4)


def test_pairs_gradient_matches_gather():
    torch.manual_seed(0)
    b = make_batch([300], in_channels=4)
    nbr = neighbor_map(b["grid_coord"], offset2batch(b["offset"]), 3)
    w = torch.randn(6, 3, 3, 3, 4, requires_grad=True)
    x = b["feat"].clone().requires_grad_()
    g = torch.randn(len(x), 6)
    (conv_gather_gemm(x, nbr, w) * g).sum().backward()
    gx, gw = x.grad.clone(), w.grad.clone()
    x.grad = w.grad = None
    (conv_pairs(x, neighbor_pairs(nbr), w) * g).sum().backward()
    assert torch.allclose(x.grad, gx, atol=1e-4) and torch.allclose(w.grad, gw, atol=1e-4)


@pytest.mark.skipif(not (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"),
                    reason="needs CUDA or TRITON_INTERPRET=1")
@pytest.mark.parametrize("cin,cout,k", [(6, 32, 5), (32, 32, 3), (48, 40, 3)])
def test_triton_conv_matches_torch_with_duplicates(cin, cout, k):
    from ptv3.kernels.subm_conv import subm_conv_triton
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    b = make_batch([500, 500], in_channels=cin)
    batch = torch.zeros(len(b["feat"]), dtype=torch.long)       # two scenes merged: coinciding voxels
    nbr = neighbor_map(b["grid_coord"], batch, k).to(device)
    x = b["feat"].to(device).requires_grad_()
    w = torch.randn(cout, k, k, k, cin, device=device, requires_grad=True)
    bias = torch.randn(cout, device=device)
    g = torch.randn(len(x), cout, device=device)
    out = subm_conv_triton(x, nbr, w, bias)
    (out * g).sum().backward()
    gx, gw = x.grad.clone(), w.grad.clone()
    x.grad = w.grad = None
    ref = conv_pairs(x, neighbor_pairs(nbr), w, bias)
    (ref * g).sum().backward()
    assert torch.allclose(out, ref, atol=1e-4)
    assert torch.allclose(gx, x.grad, atol=1e-3) and torch.allclose(gw, w.grad, atol=1e-3)
