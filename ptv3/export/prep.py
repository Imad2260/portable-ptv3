"""All the integer work of PTv3, in NumPy, so the exported network is pure float math.

`prepare(grid_coord, feat, cfg)` handles one sample (one fragment); `collate(samples, cfg)` builds
the patch tables (their size can depend on the whole batch), concatenates the samples.
Every array mirrors something PointTransformerV3.forward computes:

  feat0                     (N, Cin)  input features
  stem_nbr                  (N, 125)  neighbour map of the 5^3 stem convolution
  s{s}_nbr                  (N_s, 27) neighbour map of stage s (3^3 convolutions)
  s{s}_grid                 (N_s, 3)  voxel coordinates of stage s (relative position encoding)
  s{s}_children             (N_s, 8)  pooling: the stage s-1 points merged into each stage s point
  s{s}_parent               (N_{s-1},) unpooling: the stage s point each stage s-1 point came from
  s{s}_o{o}_k{K}_gather     (P, K)    patch table of curve o: the point in every slot
  s{s}_o{o}_k{K}_back       (N_s,)    where each point's own output sits in the flattened table
  s{s}_o{o}_k{K}_mask       (P, K)    real slots (only when a sample is shorter than K)

Empty neighbours and children are -1 here; `collate` turns them into N_s, the index of an extra
row the network appends (zeros for convolutions, -inf for max pooling). No torch import: the
runtime (ptv3.runtime) runs on ONNX Runtime alone.
"""
import numpy as np

KEY_BITS = 16


def _spread3(v):
    v = v & 0xFFFF
    v = (v | (v << 16)) & 0x0000FF0000FF
    v = (v | (v << 8)) & 0x00F00F00F00F
    v = (v | (v << 4)) & 0x0C30C30C30C3
    v = (v | (v << 2)) & 0x249249249249
    return v


def z_order(g):
    return (_spread3(g[:, 0]) << 2) | (_spread3(g[:, 1]) << 1) | _spread3(g[:, 2])


def hilbert(g, depth):
    """Skilling's algorithm; same code as ptv3.serialization.hilbert."""
    x = [g[:, 0].copy(), g[:, 1].copy(), g[:, 2].copy()]
    q = 1 << (depth - 1)
    while q > 1:
        p = q - 1
        for i in range(3):
            high = (x[i] & q) != 0
            x[0] = np.where(high, x[0] ^ p, x[0])
            t = np.where(high, 0, (x[0] ^ x[i]) & p)
            x[0] ^= t
            x[i] ^= t
        q >>= 1
    x[1] ^= x[0]
    x[2] ^= x[1]
    t = np.zeros_like(x[0])
    q = 1 << (depth - 1)
    while q > 1:
        t = np.where((x[2] & q) != 0, t ^ (q - 1), t)
        q >>= 1
    return z_order(np.stack([xi ^ t for xi in x], axis=1))


def encode(g, depth, order):
    if order.endswith("-trans"):
        g = g[:, [1, 0, 2]]
    return z_order(g) if order in ("z", "z-trans") else hilbert(g, depth)


def neighbor_map(g, k):
    """(N, k^3) row of the voxel at g + offset, -1 if empty (one sample, unique voxels first).

    Packing is additive (no field overflows), so the key of g + offset is key(g) + key(offset): in
    sorted key order each offset's queries are already sorted, and searchsorted on sorted queries
    walks the table in order instead of jumping around it (about 4x faster for the 5^3 stem).
    """
    r = k // 2
    c = g + r
    assert c.max() + r < (1 << KEY_BITS)
    pack = lambda a: (a[..., 0] << (2 * KEY_BITS)) | (a[..., 1] << KEY_BITS) | a[..., 2]   # noqa: E731
    keys = pack(c)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    offsets = np.stack(np.meshgrid(*[np.arange(-r, r + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
    found = np.empty((len(offsets), len(keys)), dtype=np.int64)       # one contiguous row per offset
    deltas = (offsets[:, 0] << (2 * KEY_BITS)) + (offsets[:, 1] << KEY_BITS) + offsets[:, 2]    # + not |: signed
    for j, d in enumerate(deltas):
        query = sorted_keys + d
        pos = np.minimum(np.searchsorted(sorted_keys, query), len(keys) - 1)
        found[j] = np.where(sorted_keys[pos] == query, order[pos], -1)
    out = np.empty((len(keys), len(offsets)), dtype=np.int64)
    out[order] = found.T
    return out


def patch_table(code, K):
    """Upstream patches for one sample: sort by code, pad to a multiple of K with the previous patch's tail."""
    n = len(code)
    order = np.argsort(code, kind="stable")
    if n >= K:
        npad = -(-n // K) * K
        pos = np.arange(npad)
        pos = np.where(pos < n, pos, pos - K)                 # borrow the points one patch earlier
        seq, mask = order[pos].reshape(-1, K), None
    else:                                                     # a sample shorter than K: one masked patch
        seq = np.full((1, K), order[0])
        seq[0, :n] = order
        mask = np.zeros((1, K), dtype=bool)
        mask[0, :n] = True
    back = np.empty(n, dtype=np.int64)
    back[order] = np.arange(n)                                # serialized rank = flat slot of the point
    return seq, mask, back


def block_tables(cfg):
    """(stage, order index, patch size) of every attention block, encoder and decoder."""
    n_orders = len(cfg["order"])
    need = set()
    for s, depth in enumerate(cfg["enc_depths"]):
        need |= {(s, i % n_orders, cfg["enc_patch_size"][s]) for i in range(depth)}
    for s, depth in enumerate(cfg["dec_depths"]):
        need |= {(s, i % n_orders, cfg["dec_patch_size"][s]) for i in range(depth)}
    return sorted(need)


def prepare(grid_coord, feat, cfg):
    """One sample -> dict of int64 / bool / float32 arrays (see module docstring).

    cfg: the model keyword arguments (a preset) plus `patch_mode` and `serialization_depth`.
    """
    g = grid_coord.astype(np.int64)
    # None (upstream: the batch's depth) can't be known per sample; it equals "sample" for one sample a run
    depth = cfg.get("serialization_depth")
    depth = max(1, int(g.max()).bit_length()) if depth in (None, "sample") else depth
    codes = np.stack([encode(g, depth, o) for o in cfg["order"]])
    out = {"feat0": feat.astype(np.float32), "stem_nbr": neighbor_map(g, 5)}
    for s in range(len(cfg["enc_depths"])):
        if s > 0:
            shift = 1 if depth >= 1 else 0                  # stride 2 = one octree level
            codes = codes >> (3 * shift)
            _, heads, cluster = np.unique(codes[0], return_index=True, return_inverse=True)
            cluster = cluster.reshape(-1)
            counts = np.bincount(cluster)
            assert counts.max() <= 8, "more than 8 children: duplicate voxels in the input"
            by_cluster = np.argsort(cluster, kind="stable")
            slot = np.arange(len(cluster)) - np.concatenate([[0], np.cumsum(counts)[:-1]])[cluster[by_cluster]]
            children = np.full((len(counts), 8), -1, dtype=np.int64)
            children[cluster[by_cluster], slot] = by_cluster
            out[f"s{s}_children"], out[f"s{s}_parent"] = children, cluster
            codes, g, depth = codes[:, heads], g[heads] >> shift, depth - shift
        out[f"s{s}_nbr"] = neighbor_map(g, 3)
        out[f"s{s}_grid"] = g
        out[f"_codes{s}"] = codes                           # patch tables are built in collate
    return out


def patch_tables(samples, cfg):
    """Add the patch tables to prepared samples. In "min" mode upstream uses one K per batch,
    K = min(shortest sample, patch size), so the tables need every sample's size."""
    for s, o, k in block_tables(cfg):
        K = min(min(len(t[f"s{s}_grid"]) for t in samples), k) if cfg.get("patch_mode", "min") == "min" else k
        for t in samples:
            seq, mask, back = patch_table(t[f"_codes{s}"][o], K)
            name = f"s{s}_o{o}_k{k}"
            t[name + "_gather"], t[name + "_back"] = seq, back
            if mask is not None:
                t[name + "_mask"] = mask


def collate(samples, cfg):
    """Build the patch tables, then concatenate the samples, shifting every index by where its
    sample starts. A sample shorter than K adds a mask; the other samples then get all-True masks.
    """
    samples = [dict(t) for t in samples]
    patch_tables(samples, cfg)
    masks = {k for t in samples for k in t if k.endswith("_mask")}
    for t in samples:
        for k in masks:
            if k not in t:
                t[k] = np.ones_like(t[k[:-len("_mask")] + "_gather"], dtype=bool)
        for k in [k for k in t if k.startswith("_codes")]:
            del t[k]
    stages = sorted({int(k[1:k.index("_")]) for k in samples[0] if k[0] == "s" and k[1].isdigit()})
    size = lambda t, s: len(t[f"s{s}_grid"])                    # noqa: E731
    total = [sum(size(t, s) for t in samples) for s in stages]
    base = [0] * len(stages)
    table_base = {}
    parts = {k: [] for k in samples[0]}
    for t in samples:
        for key, a in t.items():
            if key == "feat0" or key.endswith("_grid") or key.endswith("_mask"):
                parts[key].append(a)
                continue
            s = 0 if key == "stem_nbr" else int(key[1:key.index("_")])
            if key.endswith("_nbr"):
                a = np.where(a < 0, total[s], a + base[s])
            elif key.endswith("_children"):                       # children are stage s-1 rows
                a = np.where(a < 0, total[s - 1], a + base[s - 1])
            elif key.endswith("_parent") or key.endswith("_gather"):
                a = a + base[s]
            elif key.endswith("_back"):
                table = key[:-len("_back")]
                a = a + table_base.get(table, 0)
                table_base[table] = table_base.get(table, 0) + t[table + "_gather"].size
            parts[key].append(a)
        for s in stages:
            base[s] += size(t, s)
    # int64 on purpose: with int32 index tables ONNX Runtime's CUDA run took twice as long and no less
    # memory (docs/DEPLOY.md)
    return {k: np.concatenate(v) for k, v in parts.items()}
