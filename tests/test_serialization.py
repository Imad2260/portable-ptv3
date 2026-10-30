import importlib
import itertools
import os
import sys

import pytest
import torch

from ptv3.serialization import ORDERS, encode, hilbert, z_order


def z_order_bitwise(grid, depth):
    code = torch.zeros(len(grid), dtype=torch.long)
    for b in range(depth):
        for axis in range(3):
            code |= ((grid[:, axis] >> b) & 1) << (3 * b + 2 - axis)
    return code


def test_z_order_matches_bitwise_interleave():
    grid = torch.randint(0, 1 << 16, (5000, 3))
    assert torch.equal(z_order(grid), z_order_bitwise(grid, 16))


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_hilbert_is_a_curve(depth):
    """Every cell of the 2^depth cube gets one index, and consecutive indices are face neighbours."""
    side = 1 << depth
    grid = torch.tensor(list(itertools.product(range(side), repeat=3)))
    code = hilbert(grid, depth)
    assert torch.equal(torch.sort(code).values, torch.arange(side ** 3))
    path = grid[torch.argsort(code)]
    assert torch.all((path[1:] - path[:-1]).abs().sum(1) == 1)


@pytest.fixture(scope="module")
def official():
    """Pointcept's serialization package, from a clone of github.com/Pointcept/PointTransformerV3."""
    repo = os.environ.get("PTV3_OFFICIAL_REPO")
    if not repo:
        pytest.skip("set PTV3_OFFICIAL_REPO to a clone of the official repository")
    sys.path.insert(0, os.path.abspath(repo))
    try:
        return importlib.import_module("serialization")
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("order", ORDERS)
def test_matches_official_encoders(official, order):
    gen = torch.Generator().manual_seed(0)
    for depth in range(1, 17):
        grid = torch.randint(0, 1 << depth, (2000, 3), generator=gen)
        batch = torch.randint(0, 4, (2000,), generator=gen)
        expected = official.encode(grid, batch, depth=depth, order=order)
        assert torch.equal(encode(grid, batch, depth=depth, order=order), expected), (order, depth)
