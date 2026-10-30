# Deploying the exported model

What the ONNX runtime needs, and how to size it for a machine. The examples assume a laptop with a
16 GB GPU and 64 GB of RAM; the measurements are from [RESULTS.md](RESULTS.md) (a larger GPU, so read
times as relative, memory as absolute: ONNX Runtime allocates the same amount on any CUDA GPU).

## What runs where

```
scene (N points)
  -> NumPy: voxelize, fragments, index tables (ptv3.export.prep)            CPU, RAM
  -> ONNX Runtime: the network on those tables                              GPU, VRAM
  -> NumPy: softmax, scatter back to the N points                           CPU, RAM
```

Install: `pip install numpy onnxruntime-gpu` (CUDA; on Windows with an AMD or Intel GPU,
`onnxruntime-directml`). No PyTorch. The CUDA provider needs cuDNN 9: the `nvidia-cudnn-cu12` pip
package is enough, the runtime loads it ([`create_session`](../ptv3/runtime/onnx_infer.py)).

## Which export

```bash
python scripts/export_onnx.py --checkpoint model_best.pth --out model.onnx --fused-attention
```

`--fused-attention` uses ONNX Runtime's fused `MultiHeadAttention` kernel (fp16 attention, everything
else fp32). Measured against the standard-operator export: same Area 5 mIoU (71.64 single pass, the
same as PyTorch), 10 % faster, 20 % less GPU memory. It needs the CUDA provider; for DirectML or the
CPU provider, export without the flag.

## GPU memory: points per run

The one setting that matters is `--max-points`: how many points go through the network at once.
ONNX Runtime's GPU memory grows linearly with it:

| Points per run | standard ops | fused attention | PyTorch + Triton (for comparison) |
|---:|---:|---:|---:|
| 100k | 2.3 GB | 1.6 GB | 0.5 GB |
| 200k | 4.7 GB | 3.9 GB | 0.9 GB |
| 400k | 9.4 GB | 7.5 GB | 1.7 GB |
| 800k | 18.8 GB | 17.0 GB | 3.4 GB |

About 19 KB per point with fused attention. On a 16 GB laptop GPU, keep 2–3 GB for the display,
the driver and the CUDA context: 400k points per run (7.5 GB) leaves room for everything else;
600k is the practical ceiling. Add `--max-points 400000`.

A fragment larger than that is split into slabs along its longer horizontal axis, each with a margin
of context (default 1 m) whose predictions are discarded ([`Segmenter.slabs`](../ptv3/runtime/onnx_infer.py)).
Without the split, a large room runs as one fragment whatever `--max-points` says: the full Area 5
evaluation then peaks at 27 GB.

Whole Area 5, single pass, fused export, 400k points per run, everything in one process:

| Runtime setting | mIoU % | Process GPU memory, peak | Time |
|---|---:|---:|---:|
| no split, arena kept (largest room as one fragment) | 71.64 | 27.5 GB | 7.4 min |
| slabs, arena kept | 71.55 | 13.0 GB | 8.1 min |
| slabs, arena shrunk after every run (default) | 71.55 | **6.0 GB** | 13.0 min |

Splitting cost 0.09 mIoU here. 11 of the 68 rooms have fragments above 400k voxels (the largest
1.12 M), and they hold 35 % of the voxels. For scenes that are always larger than one run, choose
a margin that covers the context the model needs at the slab borders.

Shrinking matters because ONNX Runtime's arena keeps the blocks of every earlier run size: with
many different sizes it held 13 GB, more than twice what one run needs. With
`memory.enable_memory_arena_shrinkage` the arena hands unused blocks back after each run, so the
peak is that of the largest single run. It costs re-allocation time: 20 % per run in a sequence of
large runs, 60 % on Area 5, where slabs make many small runs. Turn it off (`--keep-arena`,
`Segmenter(shrink_arena=False)`) when the GPU has room to spare.

Why PyTorch needs 4x less: its caching allocator reuses freed blocks within a run, and the Triton
attention never materializes the (P, H, K, K) scores. If the laptop can run PyTorch with Triton
(Linux; Windows only through the unofficial `triton-windows` builds), `scripts/evaluate.py
--attn-backend triton --conv-backend triton` is 3x faster and 4x leaner than ONNX Runtime.

## ONNX Runtime settings, and why

| Setting | Effect measured at 400k points | Where |
|---|---|---|
| `enable_mem_pattern = False` | standard ops 15.0 -> 8.4 GB, fused 8.9 -> 5.3 GB (fresh process, one run size), same time. Memory patterns plan one block per run from the previous run's shapes; with input sizes that change every run the plan is mostly waste. | `create_session` |
| int64 index tables, not int32 | int32 would halve the tables, but ONNX Runtime's CUDA run took twice as long (460 vs 230 ms) for no memory saved | `ptv3.export.prep.collate` |

The budget table above is from one process that ran growing sizes one after the other, as a
deployment sees varying scenes: the arena keeps blocks from earlier runs, so it reads higher than a
fresh process at one size (fused, 400k: 7.5 vs 5.3 GB). Plan with the table.

## CPU and RAM

The index tables cost about 1.5 KB per point (the 5³ stem neighbour map is 125 int64 per point) and
take 1.5 s per 400k points on one core of a desktop CPU.
Two things keep this off the critical path:

- **Fragments share tables.** The test fragments of one augmentation hold the i-th point of every
  voxel, so they cover the same voxels: the tables are built once per batch size and only the
  features change.
- **Preparation overlaps the GPU.** For inputs that don't share tables, the next batch is prepared in
  a worker thread while the current one runs (`workers`, `lookahead`).

64 GB of RAM is far more than needed: one 400k-point batch with its tables is under 1 GB, plus the
scene itself and its (N, classes) probabilities.

## Checklist for another dataset

1. Train or fine-tune with `python -m ptv3.train`, export with `--fused-attention`.
2. Check the metadata: `ptv3_config` holds the curve order, patch sizes and serialization depth the
   preparation uses. Don't change the order after export unless you mean to (see RESULTS.md, "Curve
   order at test time").
3. Use the same voxel size, colour normalization and feature layout at inference as in training: the
   ONNX file doesn't know them.
4. Pick `--max-points` from the table above for the target GPU, and the slab margin from the scale of
   your scenes (it should cover the context the model needs at the slab borders).
5. Compare the ONNX predictions with PyTorch on a few scenes (`tests/test_export.py` does it on
   synthetic ones).
