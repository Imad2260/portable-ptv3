"""Fused RPE attention kernel vs the reference math path.

Runs on CUDA, or on CPU with TRITON_INTERPRET=1 (slow, small sizes).
"""
import os

import pytest
import torch

from ptv3.attention import RPE

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() or INTERPRET), reason="needs CUDA or TRITON_INTERPRET=1")


def reference(qkv, grid, rpe, lengths, H, scale):
    P, K, C3 = qkv.shape
    q, k, v = qkv.reshape(P, K, 3, H, -1).permute(2, 0, 3, 1, 4).unbind(0)
    s = (q * scale) @ k.transpose(-2, -1)
    if rpe is not None:
        s = s + rpe(grid)
    if lengths is not None:
        s = s.masked_fill(torch.arange(K, device=qkv.device)[None, None, None, :] >= lengths[:, None, None, None], float("-inf"))
    return (s.softmax(-1) @ v).transpose(1, 2).reshape(P * K, -1)


@pytest.mark.parametrize("use_rpe,masked,K", [(True, False, 128), (True, True, 128), (False, True, 77), (True, False, 48)])
def test_forward_and_backward(use_rpe, masked, K):
    from ptv3.kernels.rpe_attention import patch_attention_rpe
    device = "cpu" if INTERPRET else "cuda"
    torch.manual_seed(0)
    P, H, D = 3, 2, 16
    rpe = RPE(128, H).to(device) if use_rpe else None
    if rpe is not None:
        rpe.rpe_table.data.normal_(0, 0.5)
    qkv = torch.randn(P, K, 3 * H * D, device=device, requires_grad=True)
    grid = torch.randint(0, 30, (P, K, 3), device=device)
    lengths = torch.tensor([K, K // 2, 5], device=device) if masked else None
    key_mask = (torch.arange(K, device=device)[None] < lengths[:, None]) if masked else None
    g = torch.randn(P * K, H * D, device=device)
    if masked:     # slots past a short patch's end are never read back: zero their output gradient
        g = g * key_mask.reshape(-1, 1)
    scale = D ** -0.5

    out = patch_attention_rpe(qkv, grid, rpe, key_mask, H, scale)
    (out * g).sum().backward()
    gq, gt = qkv.grad.clone(), rpe.rpe_table.grad.clone() if rpe is not None else None
    qkv.grad = None
    if rpe is not None:
        rpe.rpe_table.grad = None
    ref = reference(qkv, grid, rpe, lengths, H, scale)
    (ref * g).sum().backward()
    rows = key_mask.reshape(-1) if masked else slice(None)
    assert torch.allclose(out[rows], ref[rows], atol=1e-4)
    assert torch.allclose(gq, qkv.grad, atol=1e-4)
    if rpe is not None:
        assert torch.allclose(gt, rpe.rpe_table.grad, atol=1e-3)
