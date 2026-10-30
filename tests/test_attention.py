import torch
import torch.nn as nn

from ptv3.attention import SerializedAttention, patch_layout


def upstream_padding(offset, patch_size):
    """Pointcept's get_padding_and_inverse, verbatim apart from the return value."""
    bincount = torch.diff(offset, prepend=torch.tensor([0]))
    bincount_pad = torch.div(bincount + patch_size - 1, patch_size, rounding_mode="trunc") * patch_size
    mask_pad = bincount > patch_size
    bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
    _offset = nn.functional.pad(offset, (1, 0))
    _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
    pad = torch.arange(_offset_pad[-1])
    unpad = torch.arange(_offset[-1])
    cu_seqlens = []
    for i in range(len(offset)):
        unpad[_offset[i]:_offset[i + 1]] += _offset_pad[i] - _offset[i]
        if bincount[i] != bincount_pad[i]:
            pad[_offset_pad[i + 1] - patch_size + (bincount[i] % patch_size):_offset_pad[i + 1]] = pad[
                _offset_pad[i + 1] - 2 * patch_size + (bincount[i] % patch_size):_offset_pad[i + 1] - patch_size]
        pad[_offset_pad[i]:_offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
        cu_seqlens.append(torch.arange(_offset_pad[i], _offset_pad[i + 1], step=patch_size, dtype=torch.int32))
    cu = nn.functional.pad(torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1])
    return pad, unpad, cu


def test_layout_matches_upstream_padding():
    for counts in ([300, 128, 77, 1000, 5, 256], [130], [12, 12]):
        offset = torch.cumsum(torch.tensor(counts), 0)
        K = 128
        pad, unpad, cu = upstream_padding(offset, K)
        lay = patch_layout(offset, K, "fixed")
        # every upstream patch (cu_seqlens[i]..cu_seqlens[i+1]) is one row of our table
        for p in range(len(cu) - 1):
            seq = pad[cu[p]:cu[p + 1]]
            assert torch.equal(lay.slot_pos[p, :len(seq)], seq)
            mask = lay.key_mask[p] if lay.key_mask is not None else torch.ones(K, dtype=torch.bool)
            assert int(mask.sum()) == len(seq)
        # where each serialized position's output is read back: same patch and slot as upstream
        flat_row = torch.searchsorted(cu, unpad, right=True) - 1
        assert torch.equal(lay.dense_pos, flat_row * K + (unpad - cu[flat_row]))


def test_min_mode_has_no_mask():
    offset = torch.cumsum(torch.tensor([300, 77, 1000]), 0)
    lay = patch_layout(offset, 128, "min")
    assert lay.K == 77 and lay.key_mask is None


def test_sdpa_equals_math_with_rpe_and_mask():
    torch.manual_seed(0)
    counts = [300, 40, 260]
    offset = torch.cumsum(torch.tensor(counts), 0)
    n = int(offset[-1])
    a = SerializedAttention(64, 4, 128, enable_rpe=True, attn_backend="math", patch_mode="fixed")
    b = SerializedAttention(64, 4, 128, enable_rpe=True, attn_backend="sdpa", patch_mode="fixed")
    b.load_state_dict(a.state_dict())
    order = torch.cat([torch.randperm(c) + int(o - c) for c, o in zip(counts, offset)])[None]
    inverse = torch.empty_like(order).scatter_(1, order, torch.arange(n)[None])

    def point():
        from ptv3.model import Point
        return Point(feat=torch.randn(n, 64, generator=torch.Generator().manual_seed(1)), offset=offset,
                     grid_coord=torch.randint(0, 40, (n, 3), generator=torch.Generator().manual_seed(2)),
                     serialized_order=order, serialized_inverse=inverse)

    with torch.no_grad():
        ya, yb = a(point()).feat, b(point()).feat
    assert torch.allclose(ya, yb, atol=1e-5)
