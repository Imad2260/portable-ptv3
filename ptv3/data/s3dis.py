"""S3DIS in Pointcept's preprocessed layout: <root>/Area_k/<room>/{coord,color,normal,segment}.npy.

`scripts/download.py` fetches it (Pointcept/s3dis-compressed on the Hugging Face Hub).
The transform lists reproduce configs/s3dis/semseg-pt-v3m1-0-rpe.py of Pointcept, with the colour
normalization the official checkpoints were trained with (c / 127.5 - 1). Pointcept changed it to
c / 255 in March 2025 (v1.6.0); with that, the official S3DIS checkpoint's accuracy collapses (see
docs/RESULTS.md). Set COLOR_NORM = "unit" for models trained with current Pointcept.
"""
import copy
import os
from glob import glob

import numpy as np

from . import transforms as T

CLASSES = ("ceiling", "floor", "wall", "beam", "column", "window", "door",
           "table", "chair", "sofa", "bookcase", "board", "clutter")
TRAIN_AREAS = ("Area_1", "Area_2", "Area_3", "Area_4", "Area_6")
TEST_AREA = "Area_5"
COLOR_NORM = "pm1"


def rooms(root, areas):
    areas = [areas] if isinstance(areas, str) else list(areas)
    return sorted(p for a in areas for p in glob(os.path.join(root, a, "*")) if os.path.isdir(p))


def load_room(path):
    data = {k: np.load(os.path.join(path, k + ".npy")) for k in ("coord", "color", "normal", "segment")}
    data["coord"] = data["coord"].astype(np.float32)
    data["color"] = data["color"].astype(np.float32)
    data["normal"] = data["normal"].astype(np.float32)
    data["segment"] = data["segment"].reshape(-1).astype(np.int32)
    data["name"] = os.path.basename(path)
    return data


def train_transform(grid_size=0.02, point_max=204800):
    return T.Compose([
        T.CenterShift(apply_z=True),
        T.RandomDropout(dropout_ratio=0.2, dropout_application_ratio=0.2),
        T.RandomRotate(angle=(-1, 1), axis="z", center=(0, 0, 0), p=0.5),
        T.RandomRotate(angle=(-1 / 64, 1 / 64), axis="x", p=0.5),
        T.RandomRotate(angle=(-1 / 64, 1 / 64), axis="y", p=0.5),
        T.RandomScale(scale=(0.9, 1.1)),
        T.RandomFlip(p=0.5),
        T.RandomJitter(sigma=0.005, clip=0.02),
        T.ChromaticAutoContrast(p=0.2, blend_factor=None),
        T.ChromaticTranslation(p=0.95, ratio=0.05),
        T.ChromaticJitter(p=0.95, std=0.05),
        T.GridSample(grid_size=grid_size, mode="train"),
        T.SphereCrop(sample_rate=0.6, mode="random"),
        T.SphereCrop(point_max=point_max, mode="random"),
        T.CenterShift(apply_z=False),
        T.NormalizeColor(COLOR_NORM),
    ])


def to_sample(data):
    """The model input of one sample: coord, grid_coord, feat = [color, normal] (6 channels)."""
    out = {
        "coord": data["coord"].astype(np.float32),
        "grid_coord": data["grid_coord"].astype(np.int64),
        "feat": np.concatenate([data["color"], data["normal"]], axis=1).astype(np.float32),
    }
    for k in ("segment", "index"):
        if k in data:
            out[k] = data[k]
    return out


# test-time augmentation of the official config: 5 scales, each with and without an x/y flip
TTA = [[T.RandomScale(scale=(s, s))] for s in (0.9, 0.95, 1.0, 1.05, 1.1)] + \
      [[T.RandomScale(scale=(s, s)), T.RandomFlip(p=1)] for s in (0.9, 0.95, 1.0, 1.05, 1.1)]


def test_fragments(room, grid_size=0.02, augmentations=TTA):
    """Upstream SemSegTester input: every augmented copy split into one-point-per-voxel fragments."""
    base = T.NormalizeColor(COLOR_NORM)(T.CenterShift(apply_z=True)(dict(room)))
    fragments = []
    for aug in augmentations:
        data = T.Compose(aug)(copy.deepcopy(base))
        for part in T.GridSample(grid_size=grid_size, mode="test")(data):
            fragments.append(to_sample(T.CenterShift(apply_z=False)(part)))
    return fragments


class S3DIS:
    """Training set: `loop` passes over the rooms per epoch, as upstream."""

    def __init__(self, root, areas=TRAIN_AREAS, loop=30, transform=None):
        self.paths = rooms(root, areas)
        assert self.paths, f"no rooms found under {root} for {areas}"
        self.loop = loop
        self.transform = transform or train_transform()

    def __len__(self):
        return len(self.paths) * self.loop

    def __getitem__(self, i):
        return to_sample(self.transform(load_room(self.paths[i % len(self.paths)])))
