"""Load official Pointcept checkpoints and this repository's own."""
import torch

from .model import Segmentor
from .presets import PRESETS


def load_state_dict(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}


def build(preset, **options):
    """A Segmentor with an official preset's architecture; `options` are model keyword overrides."""
    cfg = dict(PRESETS[preset])
    cfg.update(options)
    return Segmentor(**cfg)


def load(preset, path, **options):
    """Official (Pointcept) or own checkpoint -> Segmentor in eval mode. Parameter names are identical."""
    model = build(preset, **options)
    model.load_state_dict(load_state_dict(path), strict=True)
    return model.eval()
