"""Confusion matrix and IoU scores (NumPy only)."""
import numpy as np


def confusion(pred, label, num_classes, ignore_index=-1):
    keep = label != ignore_index
    return np.bincount(label[keep] * num_classes + pred[keep], minlength=num_classes ** 2).reshape(num_classes, num_classes)


def scores(conf):
    inter = np.diag(conf).astype(np.float64)
    union = conf.sum(0) + conf.sum(1) - inter
    iou = inter / np.maximum(union, 1e-10)
    present = union > 0
    return {"mIoU": float(iou.mean()), "mIoU_present": float(iou[present].mean()) if present.any() else 0.0, "mAcc": float((inter / np.maximum(conf.sum(1), 1e-10)).mean()),
            "allAcc": float(inter.sum() / max(conf.sum(), 1)), "iou": iou.tolist()}
