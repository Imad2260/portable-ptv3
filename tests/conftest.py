import numpy as np
import pytest
import torch


def surface_scene(n, seed, grid_size=0.02):
    """Points on 4 random planes, about one point per voxel of plane: voxels have real neighbours."""
    rng = np.random.default_rng(seed)
    half = 0.5 * np.sqrt(n / 4) * grid_size                       # half side of a plane
    pts = []
    for _ in range(4):
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        u = np.cross(normal, [1.0, 0, 0] if abs(normal[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        a, b = rng.uniform(-half, half, (2, n // 4, 1))
        pts.append(rng.uniform(-half, half, 3) + a * u + b * v + rng.normal(0, 0.003, (n // 4, 1)) * normal)
    coord = np.concatenate(pts).astype(np.float32)
    grid = np.floor(coord / grid_size).astype(np.int64)
    grid -= grid.min(0)
    _, keep = np.unique(grid, axis=0, return_index=True)          # one point per voxel, like GridSample
    keep = np.sort(keep)
    return coord[keep], grid[keep]


def make_batch(sizes, in_channels=6, seed=0, device="cpu"):
    coords, grids, offsets, total = [], [], [], 0
    for i, n in enumerate(sizes):
        c, g = surface_scene(n, seed + i)
        coords.append(c)
        grids.append(g)
        total += len(c)
        offsets.append(total)
    coord = torch.from_numpy(np.concatenate(coords))
    gen = torch.Generator().manual_seed(seed)
    return {
        "coord": coord.to(device),
        "grid_coord": torch.from_numpy(np.concatenate(grids)).int().to(device),
        "feat": torch.randn(len(coord), in_channels, generator=gen).to(device),
        "offset": torch.tensor(offsets).to(device),
    }


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
