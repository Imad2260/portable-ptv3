"""Serialized patch attention.

PTv3 sorts the points of every sample along a space-filling curve and cuts the sorted sequence
into patches of K points; attention runs inside each patch. A sample whose length is not a
multiple of K is padded by repeating the tail of its previous patch (those copies attend like
real points; their outputs are dropped). A sample shorter than K forms one short patch.

The official code builds the padding with a Python loop over samples and then calls either
flash_attn (no relative position encoding) or a plain matmul-softmax-matmul. Here the layout
is built with tensor ops only, and every backend reads the same dense (P, K) table:

  sdpa   torch.nn.functional.scaled_dot_product_attention, any device, RPE as an additive bias
  math   the official non-flash computation, kept as the reference
  flash  flash_attn varlen, as upstream (CUDA, fp16/bf16, no RPE)
  triton fused kernel with the RPE lookup inside (ptv3.kernels.rpe_attention), CUDA
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import flash_attn
except ImportError:
    flash_attn = None


@dataclass
class PatchLayout:
    """Dense patch table shared by every serialization order of one stage.

    K          patch size actually used
    slot_pos   (P, K) serialized position that fills each slot (a global row of serialized_order)
    key_mask   (P, K) bool, False on the empty slots of short samples; None if there are none
    dense_pos  (N,)   for serialized position j, its flat slot index p * K + k
    """
    K: int
    slot_pos: torch.Tensor
    key_mask: Optional[torch.Tensor]
    dense_pos: torch.Tensor


@torch.no_grad()
def patch_layout(offset, patch_size, patch_mode="fixed"):
    """Same patches as Pointcept's get_padding_and_inverse, without the per-sample loop.

    patch_mode "fixed": K = patch_size; samples shorter than K become one masked patch
               (upstream flash path).
               "min":   K = min(shortest sample, patch_size), so no sample is shorter than K
               (upstream non-flash path; K then depends on the other samples of the batch).
    """
    counts = torch.diff(offset, prepend=offset.new_zeros(1))           # (B,)
    starts = offset - counts                                           # (B,)
    K = int(min(int(counts.min()), patch_size)) if patch_mode == "min" else patch_size
    num_patches = torch.div(counts + K - 1, K, rounding_mode="floor").clamp_(min=1)   # (B,)
    patch_start = torch.cumsum(num_patches, 0) - num_patches                           # (B,)
    sample = torch.repeat_interleave(torch.arange(len(counts), device=offset.device), num_patches)  # (P,)
    local = (torch.arange(len(sample), device=offset.device) - patch_start[sample])[:, None] * K \
        + torch.arange(K, device=offset.device)[None, :]                # (P, K) position inside the sample
    count = counts[sample][:, None]
    # slots past the end of a long sample borrow the points one patch earlier
    src = torch.where(local < count, local, torch.where(count > K, local - K, torch.zeros_like(local)))
    slot_pos = starts[sample][:, None] + src
    key_mask = (local < count) | (count >= K)
    key_mask = None if bool(key_mask.all()) else key_mask

    pos = torch.arange(int(offset[-1]), device=offset.device)
    pos_sample = torch.repeat_interleave(torch.arange(len(counts), device=offset.device), counts)
    pos_local = pos - starts[pos_sample]
    dense_pos = (patch_start[pos_sample] + torch.div(pos_local, K, rounding_mode="floor")) * K + pos_local % K
    return PatchLayout(K, slot_pos, key_mask, dense_pos)


class RPE(nn.Module):
    """Relative position encoding of PTv3: a learned bias per head for every (dx, dy, dz) bin."""

    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, grid):
        """grid (P, K, 3) voxel coordinates of the patch slots -> bias (P, H, K, K).

        Upstream gathers the (3 * B, H) table with a (P, K, K, 3) index and sums the three axes. Here
        the three axis bins are combined into one index into a folded (B^3, H) table (B = 2 * pos_bnd + 1),
        so one gather of (P, K, K, H) suffices. It also makes the backward pass cheaper: the gradient
        additions spread over B^3 = 35,937 rows instead of piling onto 3 * 33.
        """
        B = self.rpe_num
        rel = (grid[:, :, None, :] - grid[:, None, :, :]).clamp_(-self.pos_bnd, self.pos_bnd) + self.pos_bnd
        index = (rel[..., 0] * B + rel[..., 1]) * B + rel[..., 2]                      # (P, K, K)
        t = self.rpe_table.view(3, B, self.num_heads)
        table = (t[0][:, None, None] + t[1][None, :, None] + t[2][None, None, :]).reshape(B ** 3, self.num_heads)
        bias = table.index_select(0, index.reshape(-1)).view(*index.shape, self.num_heads)   # (P, K, K, H)
        return bias.permute(0, 3, 1, 2)


class SerializedAttention(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        upcast_attention=True,
        upcast_softmax=True,
        attn_backend="sdpa",
        patch_mode="fixed",
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert attn_backend in ("sdpa", "math", "flash", "triton"), attn_backend
        if attn_backend == "flash":
            assert flash_attn is not None, "attn_backend='flash' needs the flash_attn package"
            assert not enable_rpe, "flash_attn has no relative position encoding"
        self.channels = channels
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.attn_backend = attn_backend
        self.patch_mode = patch_mode
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpe = RPE(patch_size, num_heads) if enable_rpe else None

    def tables(self, point):
        """(gather, back, layout) for this block's order, cached on the Point (shared by the stage)."""
        cache = point.setdefault("patch_cache", {})
        key = (self.patch_size, self.patch_mode)
        if key not in cache:
            cache[key] = patch_layout(point.offset, self.patch_size, self.patch_mode)
        layout = cache[key]
        okey = key + (self.order_index,)
        if okey not in cache:
            gather = point.serialized_order[self.order_index][layout.slot_pos]     # (P, K) point ids
            back = layout.dense_pos[point.serialized_inverse[self.order_index]]   # (N,) flat slot per point
            cache[okey] = (gather, back)
        return cache[okey] + (layout,)

    def forward(self, point):
        H, C = self.num_heads, self.channels
        gather, back, layout = self.tables(point)
        K = layout.K
        dropout_p = self.attn_drop if self.training else 0.0
        qkv = self.qkv(point.feat)

        if self.attn_backend == "flash":
            feat = self._flash(qkv, gather, layout, dropout_p)
        elif self.attn_backend == "triton":
            from .kernels.rpe_attention import patch_attention_rpe
            grid = point.grid_coord[gather] if self.rpe is not None else None
            feat = patch_attention_rpe(qkv[gather], grid, self.rpe, layout.key_mask, H, self.scale)
            feat = feat.reshape(-1, C)
        else:
            q, k, v = qkv[gather].reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(0)  # (P, H, K, d)
            bias = self.rpe(point.grid_coord[gather]) if self.rpe is not None else None
            if self.attn_backend == "math":
                feat = self._math(q, k, v, bias, layout.key_mask, dropout_p)
            else:
                feat = self._sdpa(q, k, v, bias, layout.key_mask, dropout_p)
            feat = feat.transpose(1, 2).reshape(-1, C)                     # (P * K, C)
        point.feat = self.proj_drop(self.proj(feat[back].to(qkv.dtype)))
        return point

    def _sdpa(self, q, k, v, bias, key_mask, dropout_p):
        if self.upcast_attention and q.dtype != torch.float32:
            # upstream computes q k^T and the softmax in fp32 under autocast; so do we
            q, k, v = q.float(), k.float(), v.float()
        mask = None
        if key_mask is not None:
            mask = key_mask[:, None, None, :]
            if bias is not None:
                mask = bias.to(q.dtype).masked_fill(~mask, float("-inf"))
        elif bias is not None:
            mask = bias.to(q.dtype)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, scale=self.scale)

    def _math(self, q, k, v, bias, key_mask, dropout_p):
        """The upstream non-flash computation, step by step."""
        dtype = v.dtype
        if self.upcast_attention:
            q, k = q.float(), k.float()
        attn = (q * self.scale) @ k.transpose(-2, -1)                     # (P, H, K, K)
        if bias is not None:
            attn = attn + bias
        if key_mask is not None:
            attn = attn.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        if self.upcast_softmax:
            attn = attn.float()
        attn = F.dropout(attn.softmax(dim=-1), dropout_p, self.training).to(dtype)
        return attn @ v

    def _flash(self, qkv, gather, layout, dropout_p):
        """flash_attn on the packed padded sequence, exactly as upstream."""
        H, C, K = self.num_heads, self.channels, layout.K
        rows = gather.reshape(-1)
        lengths = torch.full((gather.shape[0],), K, dtype=torch.int32, device=qkv.device)
        if layout.key_mask is not None:
            keep = layout.key_mask.reshape(-1)
            rows = rows[keep]
            lengths = layout.key_mask.sum(1, dtype=torch.int32)
        cu_seqlens = F.pad(torch.cumsum(lengths, 0, dtype=torch.int32), (1, 0))
        dtype = torch.bfloat16 if qkv.is_cuda and torch.cuda.is_bf16_supported() else torch.float16
        out = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv[rows].to(dtype).reshape(-1, 3, H, C // H), cu_seqlens, max_seqlen=K,
            dropout_p=dropout_p, softmax_scale=self.scale,
        ).reshape(-1, C)
        if layout.key_mask is None:
            return out
        dense = out.new_zeros(gather.numel(), C)
        dense[keep] = out
        return dense
