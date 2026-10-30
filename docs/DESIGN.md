# Design notes

How each part of the official PTv3 was replaced, and why. File references are to this repository.

## 1. Sparse convolution without spconv (`ptv3/sparse_conv.py`, `ptv3/kernels/subm_conv.py`)

PTv3 uses submanifold sparse convolutions: an output only at the input voxels,

    out[i] = bias + Σ_j  W_j · x[n(i, j)]      over the offsets j whose neighbour voxel n(i, j) exists.

**Neighbour map.** Every voxel's (batch, x, y, z) is packed into one int64 key; the keys are sorted
once, and all N · k³ neighbour queries are answered by one `torch.searchsorted` call (in chunks for
very large inputs). The result is an (N, k³) table of row indices, -1 for empty. Built once per stage
and shared by every block of the stage, like spconv's `indice_key`.

**Weights.** `SubMConv3d` stores its weight exactly as spconv does, `(C_out, k, k, k, C_in)`, with
offsets in row-major `(dx, dy, dz)` order, so official checkpoints load unchanged.

**Three ways to evaluate the sum**:

| Path | When | How |
|---|---|---|
| gather-GEMM | inference, PyTorch | append a zero row, gather (rows, k³ · C_in) in blocks of ≤ 2²⁶ elements, one GEMM per block |
| pairs | training, PyTorch | per offset, the (row, neighbour) pairs that exist: one small GEMM and an `index_put(accumulate=True)` each |
| Triton implicit GEMM | training and inference, CUDA | per (row block, output-channel block): loop over offsets, gather neighbour rows straight from memory into `tl.dot`; offsets with no neighbour in the whole row block are skipped |

The pairs path exists because the gather-GEMM's backward pass sends the gradient of every empty slot
to the one zero row; those additions serialize (seconds per layer on sparse inputs). The Triton
backward pass scatters `dY · W_jᵀ` with atomic adds, which is exact even when two rows share a voxel:
merged training samples (Mix3D-style mixing) do contain coinciding voxels, and there a "flipped
neighbour map" shortcut would be wrong.

## 2. Patch attention without flash_attn (`ptv3/attention.py`, `ptv3/kernels/rpe_attention.py`)

PTv3 sorts each sample's points along a space-filling curve and cuts the sequence into patches of K
points. A sample whose length is not a multiple of K is padded by repeating the tail of its previous
patch; a sample shorter than K forms one short patch.

**Layout.** `patch_layout` builds, with tensor operations only, the dense `(P, K)` table of serialized
positions, a key mask (only when a sample is shorter than K) and, for every point, the flat slot that
holds its output. Upstream builds the same padding with a Python loop over samples. Two patch-size
rules are supported: `fixed` (upstream's flash path) and `min` (upstream's non-flash path, where K is
the smallest sample's length, capped at the patch size).

**Backends.**

| Backend | Device | Notes |
|---|---|---|
| `math` | any | upstream's non-flash computation, kept as the reference (bit-identical to upstream) |
| `sdpa` | any | `F.scaled_dot_product_attention`; RPE and masks as an additive bias |
| `flash` | CUDA | upstream's flash_attn varlen call (no RPE) |
| `triton` | CUDA | fused kernel, below |

**Fused RPE attention.** Upstream's relative position encoding builds a `(P, K, K, 3)` index, gathers a
`(P, K, K, 3, H)` tensor and sums it to a `(P, H, K, K)` bias, and autograd keeps those tensors for
the backward pass. The Triton kernel loads a patch's voxel coordinates, computes each (query, key)
bias in registers, and never writes a K × K tensor. One program per (patch, head); K ≤ 128 fits one
tile, so the softmax needs no online rescaling. The backward pass recomputes probabilities from the
saved log-sum-exp; dQ, dK, dV need no atomics because one program owns a whole patch-head, and the RPE
table's gradient is reduced per bin inside the program, then added with one atomic per bin.

## 3. No torch_scatter

Pooling's segment max and coordinate mean use `Tensor.scatter_reduce(include_self=False)`.

## 4. Serialization (`ptv3/serialization.py`)

Same codes as upstream (checked bit for bit, depths 1–16, all four orders). z-order spreads bits with
four shift-or-mask steps instead of lookup tables; Hilbert is Skilling's algorithm on whole integers
instead of `(N, 3, depth)` boolean tensors.

## 5. ONNX export (`ptv3/export/`, `scripts/export_onnx.py`, `ptv3/runtime/`)

An ONNX graph is fixed dataflow: no custom operators, no Python integers read from data, no outputs
whose size depends on the data. PTv3's forward mixes such integer bookkeeping with float math, so the
export splits them:

- `ptv3/export/prep.py` (NumPy only): serialization codes, pooling parents and an (M, 8) children
  table, neighbour maps, patch tables, voxel coordinates per stage. `collate` concatenates samples and
  shifts every index.
- `ptv3/export/graph.py`: the forward pass on those tables, reusing the loaded model's modules (no
  weight conversion). Only Gather, MatMul, Add/Mul, LayerNorm, Gelu, Softmax, ReduceMax remain. Eval
  BatchNorm is written as a multiply-add (cuDNN's BatchNormalization refuses inputs of ~100k rows);
  gathers use `index_select` (ONNX `Gather`), not `x[idx]` (`GatherND`, much more memory in ONNX
  Runtime); wide convolutions loop over offsets so the largest temporary is (N, C_in), not (N, k³, C_in).
- The settings the preparation needs are stored in the ONNX file's metadata, so the runtime needs
  only the `.onnx` file, NumPy and ONNX Runtime (CUDA, DirectML or CPU provider).
- `--fused-attention` replaces the attention math by ONNX Runtime's `com.microsoft.MultiHeadAttention`
  (fp16, the RPE as `attention_bias`), whose fused kernels never store the (P, H, K, K) scores.
- The relative position bias is one Gather from a (33³, H) table: the three per-axis tables are
  summed into one when the model loads, and the combined bin index is computed once per (stage,
  curve) and shared by the blocks that use it. Upstream gathers three tables and sums three
  (P, K, K, H) tensors in every block.

The neighbour maps are the bulk of the preparation (the 5³ stem: 125 lookups per point). Keys pack
(x, y, z) into one integer, and packing is additive, so key(g + offset) = key(g) + key(offset): in
sorted key order every offset's queries are already sorted, and `searchsorted` on sorted queries walks
the table in order instead of jumping around it (1.7x faster preparation than unsorted queries).

What the ONNX Runtime side needed (measured in [DEPLOY.md](DEPLOY.md)):

- **Memory patterns off.** ONNX Runtime plans one block for a run from the previous run's shapes;
  with input sizes that change every run that plan costs ~40 % more device memory and no time.
- **int64 index tables.** int32 tables would be half the size, but the exporter casts every index
  back to int64 before its Gather, and with the casts removed ONNX Runtime's CUDA run took twice as
  long. int64 is faster and, measured, uses no more device memory.
- **Shared tables.** Test fragments of one augmentation (the i-th point of every voxel) cover the
  same voxels in the same order, so their index tables are identical: the runtime builds them once
  per batch size and swaps the features (Area 5, 4 rooms: 90 s -> 43 s).
- **cuDNN without PyTorch.** The CUDA provider needs cuDNN; when nothing else loaded it, the
  runtime loads it from the `nvidia-*` pip packages (ONNX Runtime >= 1.21: `ort.preload_dlls()`).

## 6. Behaviour of the official code, and the options that reproduce it

Found while matching the official checkpoint bit for bit. Preset `UPSTREAM` reproduces the current
official code; `OFFICIAL` reproduces the code the official checkpoints were trained with; `FAST` is
`OFFICIAL` with deterministic evaluation.

### The official checkpoints don't match the current official code

The checkpoints on the Hugging Face Hub were trained in December 2023. Two later upstream changes
alter what they compute, without any error:

1. **CPE input** (commit "Fix PTv3 block structure", 2024-03-26). In the December 2023 code,
   `Block.forward` ended with `point.sparse_conv_feat.replace_feature(point.feat)` without assigning the
   result. `PointSequential` copies every plain module's output into the spconv tensor, so at the end
   of a block that tensor held the output of the MLP branch (after DropPath, before the residual add),
   and the next block's CPE convolution read that, not the residual stream. The fix assigns the result,
   so current code feeds the CPE convolution the residual stream: a different network for the same
   weights. Option `cpe_input="mlp_branch"` reproduces the 2023 behaviour.
2. **Colour normalization** (Pointcept v1.6.0, 2025-03-21): `NormalizeColor` changed from
   `c / 127.5 - 1` to `c / 255`. The S3DIS loader here uses `c / 127.5 - 1` (`ptv3.data.s3dis.COLOR_NORM`).

Measured on two Area 5 rooms with one fragment each: overall accuracy 0.54 with current code
(residual CPE input, 2023 colours), 0.79 with the 2023 block but `c / 255`, 0.90 with both 2023
behaviours (the official training log reports 0.92 for these rooms with 10x test-time augmentation).
Full Area 5 numbers are in [RESULTS.md](RESULTS.md).

| Upstream behaviour | Effect | Option |
|---|---|---|
| The serialization orders are shuffled with `randperm` at test time too (in `Point.serialization` and in every pooling layer) | two runs on the same input give different logits | `shuffle_at_eval=False` (preset `FAST`) |
| Non-flash path: patch size = the smallest sample of the batch | a sample's result depends on the other samples of its batch | `patch_mode="fixed"` (with a key mask for short samples) |
| Curve depth = bit length of the batch's largest grid coordinate | the Hilbert curve of a sample changes with its batch neighbours (unless the depths differ by a multiple of 3: this Hilbert construction repeats its orientation every 3 levels) | `serialization_depth="sample"` (preset `FAST`): each sample's own depth, which is what upstream computes when it runs one fragment per pass, as in its test protocol |
| With the shuffle off, block i uses curve `order[i % 4]`, so the 2-block stages only ever use the first two curves; with the checkpoint's order (z, z-trans, hilbert, hilbert-trans) only stage 3 sees a Hilbert curve | a fixed order loses up to 0.6 mIoU against the random one (single pass, Area 5) | preset `FAST` alternates the families: (z, hilbert, z-trans, hilbert-trans), as good as the random shuffle ([RESULTS.md](RESULTS.md#curve-order-at-test-time)) |
| After unpooling, `parent.feat` is updated but the spconv tensor is not, so the first convolution of each decoder stage reads the projected skip features only | the first CPE of each decoder stage never sees the upsampled coarse features | `unpool_conv_input="sum"` (needs retraining) |
| Mix3D-style batch mixing merges two scenes under one offset without recomputing grid coordinates | voxels of the two scenes coincide in the sparse convolutions | supported: lookups resolve every voxel to one row, and the Triton backward pass is exact for that forward pass |

## 7. Determinism

With `shuffle_at_eval=False`, the PyTorch backends give identical outputs on repeated calls (tested in
`tests/test_model.py`). spconv picks its algorithm by timing, which no PyTorch flag controls; the
convolutions here have no such tuning.
