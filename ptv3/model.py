"""Point Transformer V3 (Wu et al., CVPR 2024) without spconv, torch_scatter or flash_attn.

Derived from the official implementation (github.com/Pointcept/PointTransformerV3, MIT license,
see THIRD_PARTY_NOTICES.md). Module and parameter names are unchanged, so official checkpoints load
with `ptv3.checkpoint.load`. What changed:

  - sparse convolutions are gathers + GEMMs (ptv3.sparse_conv) or a Triton implicit GEMM, no spconv
  - pooling uses torch.Tensor.scatter_reduce, no torch_scatter
  - attention backends: SDPA (default), the upstream math path, flash_attn, a fused Triton kernel
  - patch padding is built with tensor ops instead of a Python loop over samples
  - eval runs are deterministic: the random permutation of serialization orders, which upstream
    also applies at test time, is a training-only augmentation unless `shuffle_at_eval=True`

Options that reproduce upstream exactly (all defaults below are upstream's):
  serialization_depth=None   curve depth from the batch's largest grid coordinate ("sample": each
                             sample's own, so its output doesn't depend on the batch)
  patch_mode=None            "min" for the non-flash path (K = min(shortest sample, patch size)),
                             "fixed" for flash
  unpool_conv_input="skip"   upstream never refreshes the sparse tensor after unpooling, so the first
                             convolution of every decoder stage sees the projected skip features only
  shuffle_at_eval=True       upstream permutes the orders at test time too
  cpe_input="residual"       current upstream; "mlp_branch" reproduces Pointcept's Dec 2023 code, which
                             trained the official checkpoints: there, each block's CPE convolution read the
                             previous block's MLP branch output instead of the residual stream

Data flow:
    dict(coord, grid_coord, feat, offset) -> Point -> serialization (space-filling curves)
      -> embedding (5^3 sparse conv) -> encoder stages (pooling + blocks)
      -> decoder stages (unpooling + blocks) -> per-point features (64 ch) -> seg_head -> logits
"""
import math

import torch
import torch.nn as nn

from .attention import SerializedAttention
from .serialization import encode, encode_per_sample
from .sparse_conv import SubMConv3d


def segment_reduce(src, index, size, reduce):
    """Reduce the rows of `src` that share an `index` value: (N, C) -> (size, C). reduce: amax | mean."""
    out = src.new_zeros((size,) + src.shape[1:])
    index = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_reduce(0, index, src, reduce=reduce, include_self=False)


def offset2batch(offset):
    counts = torch.diff(offset, prepend=offset.new_zeros(1))
    return torch.repeat_interleave(torch.arange(len(offset), device=offset.device), counts)


def batch2offset(batch):
    return torch.cumsum(torch.bincount(batch), 0)


class Point(dict):
    """A batch of point clouds as an attribute dict.

      coord       (N, 3) float   coordinates
      grid_coord  (N, 3) int     voxel coordinates (from GridSample, per sample)
      feat        (N, C) float   features
      offset      (B,)   long    cumulative point count per sample (Pointcept convention)
      batch       (N,)   long    sample index of every point
      serialized_code / _order / _inverse  (S, N), one row per space-filling curve
      patch_cache, conv_cache              index tables shared by the blocks of one stage
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "batch" not in self and "offset" in self:
            self["batch"] = offset2batch(self["offset"])
        elif "offset" not in self and "batch" in self:
            self["offset"] = batch2offset(self["batch"])

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value

    @torch.no_grad()
    def serialization(self, order, depth=None, shuffle_orders=False):
        if "grid_coord" not in self:
            # per-sample origin: a sample's voxels must not depend on its batch neighbours
            origin = segment_reduce(self.coord, self.batch, len(self.offset), "amin")[self.batch]
            self["grid_coord"] = torch.div(self.coord - origin, self.grid_size, rounding_mode="trunc").int()
        if depth == "sample":  # each sample's own depth: a sample's curve doesn't depend on its batch
            top = segment_reduce(self.grid_coord.max(1).values, self.batch, len(self.offset), "amax")
            sample_depth = [max(1, int(m).bit_length()) for m in top.tolist()]
            depth = max(sample_depth)
            code = torch.stack([encode_per_sample(self.grid_coord, self.batch, sample_depth, depth, o) for o in order])
        else:
            if depth is None:  # upstream: adapt to the batch (so the Hilbert curve changes with it)
                depth = max(1, int(self.grid_coord.max()).bit_length())
            code = torch.stack([encode(self.grid_coord, self.batch, depth, order=o) for o in order])
        assert depth * 3 + len(self.offset).bit_length() <= 63 and depth <= 16
        self["serialized_depth"] = depth
        perm_order = torch.argsort(code)
        inverse = torch.empty_like(perm_order).scatter_(
            1, perm_order, torch.arange(code.shape[1], device=code.device).expand_as(perm_order))
        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code, perm_order, inverse = code[perm], perm_order[perm], inverse[perm]
        self["serialized_code"] = code
        self["serialized_order"] = perm_order
        self["serialized_inverse"] = inverse


class PointSequential(nn.Module):
    """Sequential container: Point modules get the Point, plain modules get point.feat."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        for i, module in enumerate(args):
            self.add_module(str(i), module)
        for name, module in kwargs.items():
            self.add_module(name, module)

    def add(self, module, name=None):
        self.add_module(name or str(len(self._modules)), module)

    def __len__(self):
        return len(self._modules)

    def forward(self, point):
        for module in self._modules.values():
            if isinstance(module, (PointModule, PointSequential)):
                point = module(point)
            elif isinstance(module, SubMConv3d):
                point.feat = module(point, point.pop("conv_input", point.feat))
            else:
                point.feat = module(point.feat)
        return point


class PointModule(nn.Module):
    """Marker: takes and returns a Point."""


class SerializedAttentionModule(SerializedAttention, PointModule):
    pass


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DropPath(nn.Module):
    """Stochastic depth: drop the residual branch per row (here: per point), as timm does."""

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep


class Block(PointModule):
    """Conditional positional encoding (3^3 sparse conv) + patch attention + MLP, pre-norm residuals."""

    def __init__(self, channels, num_heads, patch_size, mlp_ratio, qkv_bias, qk_scale, attn_drop, proj_drop,
                 drop_path, norm_layer, act_layer, pre_norm, order_index, cpe_indice_key, enable_rpe,
                 upcast_attention, upcast_softmax, attn_backend, patch_mode, cpe_input="residual"):
        super().__init__()
        assert cpe_input in ("residual", "mlp_branch")
        self.cpe_input = cpe_input
        self.channels = channels
        self.pre_norm = pre_norm
        self.cpe = PointSequential(
            SubMConv3d(channels, channels, kernel_size=3, bias=True, indice_key=cpe_indice_key),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )
        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttentionModule(
            channels=channels, patch_size=patch_size, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, order_index=order_index,
            enable_rpe=enable_rpe, upcast_attention=upcast_attention, upcast_softmax=upcast_softmax,
            attn_backend=attn_backend, patch_mode=patch_mode,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(MLP(channels, int(channels * mlp_ratio), channels, act_layer, proj_drop))
        self.drop_path = PointSequential(DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

    def forward(self, point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        if self.cpe_input == "mlp_branch":
            # Pointcept's Dec 2023 code never stored the residual sum in the spconv tensor, so the next
            # block's CPE convolution read this branch output (see docs/DESIGN.md, section 6)
            point["conv_input"] = point.feat
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        return point


class SerializedPooling(PointModule):
    """Merge the (up to 8) points that share a parent cell one octree level up; max-pool features."""

    def __init__(self, in_channels, out_channels, stride=2, norm_layer=None, act_layer=None, reduce="max",
                 shuffle_orders=True, shuffle_at_eval=False):
        super().__init__()
        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        assert reduce in ("max", "mean")
        self.stride = stride
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.shuffle_at_eval = shuffle_at_eval
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = PointSequential(norm_layer(out_channels))
        self.act = PointSequential(act_layer())

    def forward(self, point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        with torch.no_grad():
            code = point.serialized_code >> pooling_depth * 3
            _, cluster, counts = torch.unique(code[0], sorted=True, return_inverse=True, return_counts=True)
            # first point of every cluster: all of a cluster's points share code, grid cell and batch
            head = torch.full((len(counts),), len(cluster), device=cluster.device).scatter_reduce_(
                0, cluster, torch.arange(len(cluster), device=cluster.device), "amin")
            code = code[:, head]
            order = torch.argsort(code)
            inverse = torch.empty_like(order).scatter_(
                1, order, torch.arange(code.shape[1], device=code.device).expand_as(order))
            if self.shuffle_orders and (self.training or self.shuffle_at_eval):
                perm = torch.randperm(code.shape[0])
                code, order, inverse = code[perm], order[perm], inverse[perm]
        n = len(counts)
        child = Point(
            feat=segment_reduce(self.proj(point.feat), cluster, n, "a" + self.reduce if self.reduce == "max" else "mean"),
            coord=segment_reduce(point.coord, cluster, n, "mean"),
            grid_coord=point.grid_coord[head] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head],
            pooling_inverse=cluster,
            pooling_parent=point,
        )
        return self.act(self.norm(child))


class SerializedUnpooling(PointModule):
    def __init__(self, in_channels, skip_channels, out_channels, norm_layer, act_layer, conv_input="skip"):
        super().__init__()
        assert conv_input in ("skip", "sum")
        self.conv_input = conv_input
        self.proj = PointSequential(nn.Linear(in_channels, out_channels), norm_layer(out_channels), act_layer())
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels), norm_layer(out_channels), act_layer())

    def forward(self, point):
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        if self.conv_input == "skip":
            parent["conv_input"] = parent.feat       # what upstream's stale sparse tensor holds
        else:
            parent.pop("conv_input", None)           # nothing stale may reach the decoder's first conv
        parent.feat = parent.feat + point.feat[inverse]
        return parent


class Embedding(PointModule):
    def __init__(self, in_channels, embed_channels, norm_layer, act_layer):
        super().__init__()
        self.stem = PointSequential(
            conv=SubMConv3d(in_channels, embed_channels, kernel_size=5, bias=False, indice_key="stem"),
            norm=norm_layer(embed_channels),
            act=act_layer(),
        )

    def forward(self, point):
        return self.stem(point)


class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        # options added by this repository
        attn_backend="sdpa",
        conv_backend="torch",
        patch_mode=None,
        serialization_depth=None,
        unpool_conv_input="skip",
        cpe_input="residual",
        shuffle_at_eval=True,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else list(order)
        self.shuffle_orders = shuffle_orders
        self.shuffle_at_eval = shuffle_at_eval
        self.serialization_depth = serialization_depth
        patch_mode = patch_mode or ("fixed" if enable_flash else "min")
        assert self.num_stages == len(stride) + 1 == len(enc_channels) == len(enc_num_head) == len(enc_patch_size)
        assert self.num_stages == len(dec_depths) + 1 == len(dec_channels) + 1 == len(dec_num_head) + 1
        bn_layer = lambda c: nn.BatchNorm1d(c, eps=1e-3, momentum=0.01)  # noqa: E731
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        def block(s, i, channels, heads, patch, dpr):
            return Block(channels, heads, patch, mlp_ratio, qkv_bias, qk_scale, attn_drop, proj_drop, dpr,
                         ln_layer, act_layer, pre_norm, i % len(self.order), f"stage{s}", enable_rpe,
                         upcast_attention, upcast_softmax, attn_backend, patch_mode, cpe_input)

        self.embedding = Embedding(in_channels, enc_channels[0], bn_layer, act_layer)
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            dpr = enc_dpr[sum(enc_depths[:s]):sum(enc_depths[:s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(SerializedPooling(enc_channels[s - 1], enc_channels[s], stride[s - 1], bn_layer, act_layer,
                                          shuffle_at_eval=shuffle_at_eval), name="down")
            for i in range(enc_depths[s]):
                enc.add(block(s, i, enc_channels[s], enc_num_head[s], enc_patch_size[s], dpr[i]), name=f"block{i}")
            self.enc.add(enc, name=f"enc{s}")

        dec_dpr = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
        self.dec = PointSequential()
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        for s in reversed(range(self.num_stages - 1)):
            dpr = dec_dpr[sum(dec_depths[:s]):sum(dec_depths[:s + 1])][::-1]
            dec = PointSequential()
            dec.add(SerializedUnpooling(dec_channels[s + 1], enc_channels[s], dec_channels[s], bn_layer, act_layer,
                                        unpool_conv_input), name="up")
            for i in range(dec_depths[s]):
                dec.add(block(s, i, dec_channels[s], dec_num_head[s], dec_patch_size[s], dpr[i]), name=f"block{i}")
            self.dec.add(dec, name=f"dec{s}")
        self.set_conv_backend(conv_backend)

    def set_conv_backend(self, backend):
        """"torch" (any device) or "triton" (CUDA): see ptv3.sparse_conv."""
        for m in self.modules():
            if isinstance(m, SubMConv3d):
                m.backend = backend

    def forward(self, data_dict):
        point = data_dict if isinstance(data_dict, Point) else Point(data_dict)
        point.serialization(self.order, depth=self.serialization_depth,
                            shuffle_orders=self.shuffle_orders and (self.training or self.shuffle_at_eval))
        point = self.embedding(point)
        point = self.enc(point)
        point = self.dec(point)
        return point


class Segmentor(nn.Module):
    """Backbone + linear head, with Pointcept's DefaultSegmentorV2 parameter names."""

    def __init__(self, num_classes, backbone_out_channels=64, **backbone_kwargs):
        super().__init__()
        self.backbone = PointTransformerV3(**backbone_kwargs)
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, data_dict):
        """dict(coord, grid_coord, feat, offset) -> (N, num_classes) logits."""
        return self.seg_head(self.backbone(data_dict).feat)
