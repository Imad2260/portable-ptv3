"""Cross-entropy + Lovasz-softmax, the PTv3 segmentation loss."""
import torch
import torch.nn.functional as F


def lovasz_softmax(logits, labels, ignore_index=-1):
    """Lovasz-softmax over the classes present in `labels` (Berman et al., CVPR 2018).

    Same value as Pointcept's LovaszLoss(mode="multiclass"), which sorts the errors of one class
    at a time in a Python loop; here every class column is sorted by one torch.sort call.
    """
    keep = labels != ignore_index
    probs = logits[keep].float().softmax(dim=1)                        # (P, C)
    labels = labels[keep].long()
    if probs.numel() == 0:
        return logits.sum() * 0.0
    present = torch.unique(labels)
    fg = (labels[:, None] == present[None, :]).float()                 # (P, Cp) one column per present class
    errors = (fg - probs[:, present]).abs()
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    fg_sorted = torch.gather(fg, 0, perm)
    gts = fg_sorted.sum(0)
    intersection = gts - fg_sorted.cumsum(0)
    union = gts + (1 - fg_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[:-1]                           # gradient of the Lovasz extension
    return (errors_sorted * jaccard).sum(0).mean()


class SegLoss(torch.nn.Module):
    def __init__(self, ce_weight=1.0, lovasz_weight=1.0, ignore_index=-1):
        super().__init__()
        self.ce_weight, self.lovasz_weight, self.ignore_index = ce_weight, lovasz_weight, ignore_index

    def forward(self, logits, labels):
        labels = labels.long()
        ce = F.cross_entropy(logits.float(), labels, ignore_index=self.ignore_index)
        return self.ce_weight * ce + self.lovasz_weight * lovasz_softmax(logits, labels, self.ignore_index)
