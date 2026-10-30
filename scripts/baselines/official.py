"""The official PTv3 model and loss, loaded from a clone of github.com/Pointcept/PointTransformerV3.

    git clone https://github.com/Pointcept/PointTransformerV3 third_party/PointTransformerV3

The official model needs spconv, torch_scatter, timm, addict (and flash_attn for flash configs).
`pointcept_lovasz` is Pointcept's multiclass Lovasz-softmax (pointcept/models/losses/lovasz.py, MIT),
with the registry code removed, so the training benchmark runs the upstream loss as well.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def official_segmentor(repo, preset_cfg, state_dict=None):
    """Official backbone + linear head (Pointcept's DefaultSegmentorV2) as one module."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(repo)))
    module = __import__(os.path.basename(os.path.abspath(repo)) + ".model", fromlist=["PointTransformerV3"])
    backbone_cfg = {k: v for k, v in preset_cfg.items() if k not in ("num_classes", "backbone_out_channels")}

    class Segmentor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = module.PointTransformerV3(**backbone_cfg)
            self.seg_head = nn.Linear(preset_cfg["backbone_out_channels"], preset_cfg["num_classes"])

        def forward(self, data):
            return self.seg_head(self.backbone(data).feat)

    model = Segmentor()
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model


def _lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if len(gt_sorted) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[0:-1]
    return jaccard


def pointcept_lovasz(logits, labels, ignore_index=-1):
    probas = logits.softmax(dim=1)
    valid = labels != ignore_index
    probas, labels = probas[valid], labels[valid]
    losses = []
    for c in labels.unique():
        fg = (labels == c).type_as(probas)
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm.data])))
    return sum(losses) / len(losses)


def pointcept_loss(logits, labels):
    return F.cross_entropy(logits, labels.long(), ignore_index=-1) + pointcept_lovasz(logits, labels.long())
