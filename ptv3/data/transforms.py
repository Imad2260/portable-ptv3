"""Point cloud transforms with the semantics of Pointcept's (pointcept/datasets/transform.py).

Every transform takes and returns a dict of NumPy arrays (coord, color, normal, segment, ...).
Random draws use NumPy's and Python's global generators in the same order as upstream, so a
seeded run makes the same choices. Only what the PTv3 S3DIS / ScanNet recipes use is here.
"""
import random

import numpy as np

POINT_KEYS = ("coord", "color", "normal", "segment", "instance", "origin_coord", "origin_segment", "strength")


def index(data, idx, keys=POINT_KEYS):
    """Keep the points `idx` in every per-point array."""
    out = dict(data)
    for k in keys:
        if k in out:
            out[k] = out[k][idx]
    return out


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class CenterShift:
    """Move the bounding-box centre to x = y = 0 (and the floor to z = 0 with apply_z)."""

    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data):
        lo, hi = data["coord"].min(0), data["coord"].max(0)
        shift = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] if self.apply_z else 0]
        data["coord"] = data["coord"] - np.asarray(shift, dtype=data["coord"].dtype)
        return data


class NormalizeColor:
    """Colour to model input. "pm1": c / 127.5 - 1 (Pointcept until March 2025, what the official PTv3
    checkpoints were trained with); "unit": c / 255 (Pointcept since v1.6.0)."""

    def __init__(self, mode="pm1"):
        assert mode in ("pm1", "unit")
        self.mode = mode

    def __call__(self, data):
        if "color" in data:
            data["color"] = data["color"] / 127.5 - 1 if self.mode == "pm1" else data["color"] / 255
        return data


class RandomDropout:
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        self.ratio = dropout_ratio
        self.p = dropout_application_ratio

    def __call__(self, data):
        if random.random() < self.p:
            n = len(data["coord"])
            data = index(data, np.random.choice(n, int(n * (1 - self.ratio)), replace=False))
        return data


class RandomRotate:
    def __init__(self, angle=(-1, 1), axis="z", center=None, p=0.5, always_apply=False):
        self.angle, self.axis, self.center = angle, axis, center
        self.p = 1 if always_apply else p

    def __call__(self, data):
        if random.random() > self.p:
            return data
        a = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        c, s = np.cos(a), np.sin(a)
        rot = {"x": [[1, 0, 0], [0, c, -s], [0, s, c]],
               "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
               "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[self.axis]
        rot = np.array(rot)
        if self.center is None:
            lo, hi = data["coord"].min(0), data["coord"].max(0)
            center = (lo + hi) / 2
        else:
            center = np.asarray(self.center)
        data["coord"] = (data["coord"] - center) @ rot.T + center
        if "normal" in data:
            data["normal"] = data["normal"] @ rot.T
        return data


class RandomScale:
    def __init__(self, scale=(0.95, 1.05), anisotropic=False):
        self.scale, self.anisotropic = scale, anisotropic

    def __call__(self, data):
        data["coord"] = data["coord"] * np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        return data


class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        for axis in (0, 1):
            if np.random.rand() < self.p:
                data["coord"] = data["coord"].copy()
                data["coord"][:, axis] *= -1
                if "normal" in data:
                    data["normal"] = data["normal"].copy()
                    data["normal"][:, axis] *= -1
        return data


class RandomJitter:
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma, self.clip = sigma, clip

    def __call__(self, data):
        jitter = np.clip(self.sigma * np.random.randn(len(data["coord"]), 3), -self.clip, self.clip)
        data["coord"] = data["coord"] + jitter
        return data


class ChromaticAutoContrast:
    def __init__(self, p=0.2, blend_factor=None):
        self.p, self.blend_factor = p, blend_factor

    def __call__(self, data):
        if "color" in data and np.random.rand() < self.p:
            lo = data["color"].min(0, keepdims=True)
            hi = data["color"].max(0, keepdims=True)
            diff = hi - lo
            if not np.any(diff > 0):
                return data
            scale = np.divide(255, diff, out=np.ones_like(diff, dtype=data["color"].dtype), where=diff > 0)
            contrast = (data["color"][:, :3] - lo) * scale
            blend = np.random.rand() if self.blend_factor is None else self.blend_factor
            data["color"][:, :3] = (1 - blend) * data["color"][:, :3] + blend * contrast
        return data


class ChromaticTranslation:
    def __init__(self, p=0.95, ratio=0.05):
        self.p, self.ratio = p, ratio

    def __call__(self, data):
        if "color" in data and np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data["color"][:, :3] = np.clip(tr + data["color"][:, :3], 0, 255)
        return data


class ChromaticJitter:
    def __init__(self, p=0.95, std=0.005):
        self.p, self.std = p, std

    def __call__(self, data):
        if "color" in data and np.random.rand() < self.p:
            noise = np.random.randn(len(data["color"]), 3) * self.std * 255
            data["color"][:, :3] = np.clip(noise + data["color"][:, :3], 0, 255)
        return data


def fnv_hash(arr):
    """FNV-1a 64-bit hash of every row of a non-negative int array, as upstream GridSample."""
    arr = arr.astype(np.uint64, copy=False)
    h = np.full(len(arr), 14695981039346656037, dtype=np.uint64)
    for j in range(arr.shape[1]):
        h *= np.uint64(1099511628211)
        h ^= arr[:, j]
    return h


class GridSample:
    """Voxelize at `grid_size`.

    train: keep one random point per voxel; adds grid_coord (voxel index from the sample's minimum).
    test:  return a list of fragments; fragment i holds point i % count of every voxel, so every
           point is in at least one fragment and every fragment has one point per voxel.
    """

    def __init__(self, grid_size=0.05, mode="train", keys=None, return_grid_coord=True, return_inverse=False):
        assert mode in ("train", "test")
        self.grid_size, self.mode, self.return_inverse = grid_size, mode, return_inverse

    def voxels(self, coord):
        scaled = coord / np.array(self.grid_size)
        grid = np.floor(scaled).astype(int)
        grid -= grid.min(0)
        key = fnv_hash(grid)
        idx_sort = np.argsort(key)
        _, inverse, count = np.unique(key[idx_sort], return_inverse=True, return_counts=True)
        return grid, idx_sort, inverse, count

    def __call__(self, data):
        grid, idx_sort, inverse, count = self.voxels(data["coord"])
        start = np.cumsum(np.insert(count, 0, 0)[:-1])
        if self.mode == "train":
            pick = idx_sort[start + np.random.randint(0, count.max(), count.size) % count]
            out = index(data, pick)
            out["grid_coord"] = grid[pick]
            if self.return_inverse:            # every original point -> the row of its voxel's representative
                cell = np.empty_like(inverse)
                cell[idx_sort] = inverse
                out["inverse"] = cell
            return out
        parts = []
        for i in range(count.max()):
            pick = idx_sort[start + i % count]
            part = index(data, pick)
            part["index"] = pick
            part["grid_coord"] = grid[pick]
            parts.append(part)
        return parts


class SphereCrop:
    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        assert mode in ("random", "center")
        self.point_max, self.sample_rate, self.mode = point_max, sample_rate, mode

    def __call__(self, data):
        n = len(data["coord"])
        point_max = int(self.sample_rate * n) if self.sample_rate is not None else self.point_max
        if n > point_max:
            center = data["coord"][np.random.randint(n) if self.mode == "random" else n // 2]
            keep = np.argsort(np.sum(np.square(data["coord"] - center), 1))[:point_max]
            data = index(data, keep, POINT_KEYS + ("grid_coord",))
        return data
