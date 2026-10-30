"""Point Transformer V3 without spconv, torch_scatter or flash_attn.

The model lives in ptv3.model (needs PyTorch); ptv3.runtime runs exported models with ONNX Runtime
and NumPy only, so this package does not import torch at import time.
"""
__version__ = "0.1.0"
