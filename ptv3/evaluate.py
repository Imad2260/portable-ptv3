"""Semantic segmentation evaluation on whole scenes.

Protocols:
  official  Pointcept's SemSegTester: 10 test-time augmentations (5 scales x flip); each augmented
            copy is voxelized into fragments that together cover every point; softmax
            probabilities are summed per point. Upstream runs one fragment per forward pass.
  single    the same fragments, without test-time augmentation
  fast      one fragment (one point per voxel); every other point takes the prediction of the
            point kept in its voxel

Fragments of one augmentation are batched into one forward pass up to `max_points` points. Only
fragments of the same augmentation share a batch: they cover the same voxels, so upstream's
batch-dependent choices (curve depth, patch size) are the same as for a single fragment.
"""
import time

import numpy as np
import torch

from .data import s3dis
from .data import transforms as T
from .metrics import confusion, scores


def collate(samples, device):
    out = {k: torch.from_numpy(np.concatenate([s[k] for s in samples])).to(device, non_blocking=True)
           for k in ("coord", "grid_coord", "feat")}
    out["offset"] = torch.tensor(np.cumsum([len(s["coord"]) for s in samples]), device=device)
    return out


def _keep(item):
    """collate_fn for batch_size=None: keep NumPy arrays (the default would turn them into tensors)."""
    return item


class RoomFragments(torch.utils.data.Dataset):
    """One item = one room under one test-time augmentation: its fragments (and the room's labels).

    Items of a room are consecutive, so the evaluation loop sums them and scores the room when the
    next room starts. One augmentation per item keeps a worker's memory to one augmented copy of a
    room, instead of ten (a large S3DIS hallway with all ten copies took 19 GB).
    """

    def __init__(self, paths, protocol, grid_size):
        self.paths, self.protocol, self.grid_size = paths, protocol, grid_size
        self.augs = s3dis.TTA if protocol == "official" else [s3dis.TTA[2]]
        self.per_room = 1 if protocol == "fast" else len(self.augs)

    def __len__(self):
        return len(self.paths) * self.per_room

    def __getitem__(self, i):
        r, a = divmod(i, self.per_room)
        room = s3dis.load_room(self.paths[r])
        if self.protocol == "fast":
            base = T.NormalizeColor(s3dis.COLOR_NORM)(T.CenterShift(apply_z=True)(dict(room)))
            part = T.GridSample(self.grid_size, mode="train", return_inverse=True)(base)
            inverse = part.pop("inverse")
            return r, room["name"], room["segment"], [s3dis.to_sample(T.CenterShift(apply_z=False)(part))], inverse
        return r, room["name"], room["segment"], s3dis.test_fragments(room, self.grid_size, [self.augs[a]]), None


@torch.no_grad()
def evaluate(model, root, area=s3dis.TEST_AREA, protocol="official", num_classes=13, grid_size=0.02,
             max_points=1_500_000, amp_dtype=None, workers=4, limit=None, log=print, device="cuda"):
    """Returns dict(mIoU, mAcc, allAcc, iou, seconds, forward_seconds, peak_gpu_gb, per_room)."""
    model.eval()
    paths = s3dis.rooms(root, area)[:limit]
    loader = torch.utils.data.DataLoader(RoomFragments(paths, protocol, grid_size), batch_size=None, collate_fn=_keep,
                                         num_workers=workers, prefetch_factor=1 if workers else None)
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start, t_forward, per_room = time.time(), 0.0, {}
    current, probs = None, None

    def finish(name, segment, probs):
        nonlocal conf
        c = confusion(probs.argmax(1).cpu().numpy(), segment.astype(np.int64), num_classes)
        conf += c
        sc = scores(c)
        per_room[name] = sc["mIoU_present"]
        log(f"{name:24s} {len(segment):>9,d} points  room mIoU {sc['mIoU_present']:.4f} (classes present)  acc {sc['allAcc']:.4f}  "
            f"running mIoU {scores(conf)['mIoU']:.4f}")

    for room_idx, name, segment, fragments, inverse in loader:
        if room_idx != current:
            if current is not None:
                finish(*last_room, probs)
            current, last_room = room_idx, (name, segment)
            probs = torch.zeros(len(segment), num_classes, device=device)
        i = 0
        while i < len(fragments):                           # pack fragments up to max_points
            j, n = i, 0
            while j < len(fragments) and (j == i or n + len(fragments[j]["coord"]) <= max_points):
                n += len(fragments[j]["coord"])
                j += 1
            batch = collate(fragments[i:j], device)
            t = time.time()
            with torch.autocast(device_type=device, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                logits = model(batch)
            prob = logits.float().softmax(-1)
            if inverse is None:
                index = torch.from_numpy(np.concatenate([f["index"] for f in fragments[i:j]])).to(device)
                probs.index_add_(0, index, prob)
            else:
                probs = prob[torch.from_numpy(inverse).to(device)]
            if device == "cuda":
                torch.cuda.synchronize()
            t_forward += time.time() - t
            i = j
    if current is not None:
        finish(*last_room, probs)
    out = scores(conf)
    out.update(seconds=time.time() - t_start, forward_seconds=t_forward, per_room=per_room,
               peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None)
    return out
