"""PTv3 forward pass on precomputed index tables (ptv3.export.prep), for ONNX export.

Only standard operators remain: Gather, MatMul, Add/Mul, LayerNorm, Gelu, Softmax, ReduceMax, Clip.
It wraps a loaded Segmentor and reuses its modules, so no weight conversion is needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.library.custom_op("ptv3::mha", mutates_args=())
def mha(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, num_heads: int, scale: float) -> torch.Tensor:
    """Attention on (B, S, C) tensors (heads packed in C) with an additive (B, H, S, S) bias.

    In PyTorch this is SDPA. scripts/export_onnx.py --fused-attention turns it into ONNX Runtime's
    com.microsoft.MultiHeadAttention, whose fused CUDA kernels never store the S x S scores.
    """
    b, n, c = q.shape
    split = lambda x: x.reshape(b, n, num_heads, c // num_heads).transpose(1, 2)   # noqa: E731
    out = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=bias.to(q.dtype), scale=scale)
    return out.transpose(1, 2).reshape(b, n, c)


@mha.register_fake
def _(q, k, v, bias, num_heads, scale):
    return torch.empty_like(q)


def take(x, idx):
    """x[idx] for an index tensor of any shape, as index_select on a flat index.

    x[idx] exports as ONNX GatherND, which ONNX Runtime runs with far more memory than Gather.
    """
    return x.index_select(0, idx.reshape(-1)).reshape(*idx.shape, *x.shape[1:])


def gather_conv(x, nbr, conv):
    """Submanifold conv; nbr maps empty neighbours to row N, appended here as zeros.

    Narrow inputs (the 6-channel stem): gather every neighbour at once, one GEMM. Wide inputs: one
    offset at a time, so the largest temporary is (N, Cin) instead of (N, k^3, Cin).
    """
    x = torch.cat([x, x.new_zeros(1, x.shape[1])])
    w = conv.weight.flatten(1, 3)                                    # (Cout, k^3, Cin)
    if w.shape[1] * w.shape[2] <= 1024:
        out = take(x, nbr).flatten(1) @ w.flatten(1).t()
    else:
        out = take(x, nbr[:, 0]) @ w[:, 0].t()
        for j in range(1, w.shape[1]):
            out = out + take(x, nbr[:, j]) @ w[:, j].t()
    return out if conv.bias is None else out + conv.bias


def batch_norm(bn, x):
    """Eval BatchNorm as a per-channel multiply-add (cuDNN's BatchNormalization refuses ~100k-row inputs)."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return x * scale + (bn.bias - bn.running_mean * scale)


def run(seq, x):
    for module in seq._modules.values():
        x = batch_norm(module, x) if isinstance(module, nn.BatchNorm1d) else module(x)
    return x


class ExportGraph(nn.Module):
    def __init__(self, segmentor, input_names, fused_attention=False):
        super().__init__()
        self.fused_attention = fused_attention
        self.m = segmentor.eval()
        self.bb = segmentor.backbone
        assert self.bb.enc._modules["enc0"]._modules["block0"].pre_norm, "export assumes pre_norm=True"
        self.input_names = list(input_names)

    def forward(self, *tensors):
        t = dict(zip(self.input_names, tensors))
        t["rpe_index"] = {}                          # (stage, order) -> combined RPE bin of every slot pair
        stem = self.bb.embedding.stem
        x = run(nn.Sequential(stem.norm, stem.act), gather_conv(t["feat0"], t["stem_nbr"], stem.conv))
        skips, conv_input = [], x
        for s, enc in enumerate(self.bb.enc._modules.values()):
            for name, module in enc._modules.items():
                if name == "down":
                    skips.append(x)
                    x = self.pool(module, x, t[f"s{s}_children"])
                else:
                    x, conv_input = self.block(module, x, conv_input if name != "block0" else x, t, s)
        n = len(self.bb.enc._modules)
        for s in reversed(range(n - 1)):
            dec = self.bb.dec._modules[f"dec{s}"]
            up = dec._modules["up"]
            x, conv_input = self.unpool(up, x, skips.pop(), t[f"s{s + 1}_parent"])
            for name, module in dec._modules.items():
                if name != "up":
                    x, conv_input = self.block(module, x, conv_input, t, s)
        return self.m.seg_head(x)

    def block(self, blk, x, conv_input, t, s):
        cpe = blk.cpe._modules
        x = x + cpe["2"](cpe["1"](gather_conv(conv_input, t[f"s{s}_nbr"], cpe["0"])))
        a = blk.attn
        name = f"s{s}_o{a.order_index}_k{a.patch_size}"
        h = blk.norm1._modules["0"](x)
        x = x + a.proj(self.attention(a, h, t[name + "_gather"], t[name + "_back"], t.get(name + "_mask"),
                                      t[f"s{s}_grid"], t["rpe_index"], (s, a.order_index, a.patch_size), self.fused_attention))
        branch = blk.mlp._modules["0"](blk.norm2._modules["0"](x))
        x = x + branch
        return x, (branch if blk.cpe_input == "mlp_branch" else x)      # what the next block's conv reads

    @staticmethod
    def rpe_bias(r, grid, gather, cache, key, dtype=torch.float32):
        """(P, H, K, K) relative position bias.

        Upstream gathers the table once per axis and sums three (P, K, K, H) tensors in every block.
        Here the three axis bins are combined into one index into a (B^3, H) table (B = 2 * pos_bnd + 1,
        the sum precomputed; ONNX Runtime folds it into a constant when it loads the model), and that
        index is computed once per (stage, order) and shared by the blocks that use it.
        """
        if key not in cache:
            g = take(grid, gather)                                   # (P, K, 3)
            rel = (g[:, :, None, :] - g[:, None, :, :]).clamp(-r.pos_bnd, r.pos_bnd) + r.pos_bnd
            cache[key] = (rel[..., 0] * r.rpe_num + rel[..., 1]) * r.rpe_num + rel[..., 2]
        t = r.rpe_table.view(3, r.rpe_num, -1)
        table = (t[0][:, None, None, :] + t[1][None, :, None, :] + t[2][None, None, :, :]).reshape(r.rpe_num ** 3, -1)
        return take(table.to(dtype), cache[key]).permute(0, 3, 1, 2)

    def attention(self, a, x, gather, back, mask, grid, cache, key, fused=False):
        H, C = a.num_heads, a.channels
        K = gather.shape[1]
        qkv = take(a.qkv(x), gather)                                 # (P, K, 3C)
        if fused:
            # fp16 on purpose: ONNX Runtime runs its fused attention kernels on fp16 inputs only
            q, k, v = (t.half() for t in qkv.reshape(-1, K, 3, C).unbind(2))     # (P, K, C) each
            # the bias is gathered from an fp16 table: no fp32 (P, H, K, K) temporary
            bias = (ExportGraph.rpe_bias(a.rpe, grid, gather, cache, key, torch.float16) if a.rpe is not None
                    else q.new_zeros(1, 1, K, K))
            if mask is not None:
                bias = bias.masked_fill(~mask[:, None, None, :], -30000.0)
            out = torch.ops.ptv3.mha(q, k, v, bias, H, a.scale).float()
            return take(out.reshape(-1, C), back)
        q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(0)
        s = (q * a.scale) @ k.transpose(-2, -1)                      # (P, H, K, K)
        if a.rpe is not None:
            s = s + self.rpe_bias(a.rpe, grid, gather, cache, key)
        if mask is not None:
            s = s.masked_fill(~mask[:, None, None, :], float("-inf"))
        out = (s.softmax(-1) @ v).transpose(1, 2).reshape(-1, C)     # (P * K, C)
        return take(out, back)

    @staticmethod
    def pool(down, x, children):
        h = down.proj(x)
        h = torch.cat([h, h.new_full((1, h.shape[1]), float("-inf"))])   # row N never wins the max
        return run(down.act, run(down.norm, take(h, children).amax(dim=1)))

    @staticmethod
    def unpool(up, x, skip, parent):
        x = run(up.proj, x)
        skip = run(up.proj_skip, skip)
        out = skip + take(x, parent)
        return out, (skip if up.conv_input == "skip" else out)
