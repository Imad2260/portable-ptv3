# Results

All numbers measured by the scripts in `scripts/`, on one GPU with nothing else running; raw
outputs are not kept in the repository. Weights: the official PTv3 S3DIS checkpoint
(`Pointcept/PointTransformerV3`, `s3dis-semseg-pt-v3m1-0-rpe`). Data: S3DIS Area 5 (68 rooms,
78.7 M points), Pointcept's preprocessed copy.

## Accuracy and evaluation time, S3DIS Area 5

Official protocol: every point covered by fragments, 10 test-time augmentations (5 scales x flip),
probabilities summed (Pointcept's `SemSegTester`). The official training log of this checkpoint
reports 73.58 mIoU. Wall time includes data loading; this repo batches fragments up to 1.5 M points
per forward pass, the official code runs one fragment per pass.

| Run | mIoU % | allAcc % | Wall time | Speed-up | Peak GPU memory | GPU forward time |
|---|---:|---:|---:|---:|---:|---:|
| official code (Dec 2023 release, the checkpoint's code) | 73.53 | 91.65 | 73.7 min | 1.0x | 32.2 GB | 68.7 min |
| this repo, Triton kernels, fp32 | 73.47 | 91.66 | 17.0 min | 4.3x | 6.9 GB | 11.1 min |
| this repo, Triton kernels, bf16 | 73.48 | 91.66 | 13.1 min | 5.6x | 5.2 GB | 7.1 min |

## Single pass (no test-time augmentation)

One augmentation (scale 1, no flip), every point covered. ONNX rows: NumPy preparation + ONNX
Runtime only; GPU memory is what the process holds (ONNX Runtime keeps its peak). Without
splitting, the largest room runs as one fragment, whatever --max-points says. The last two rows run the same
checkpoint with the current official code: see the README, "The official checkpoints no longer
match the official code".

| Run | mIoU % | allAcc % | Wall time | Speed-up | Peak GPU memory | GPU forward time |
|---|---:|---:|---:|---:|---:|---:|
| official code (Dec 2023 release) | 71.55 | 91.04 | 7.4 min | 1.0x | 27.2 GB | 6.8 min |
| this repo, Triton kernels, fp32 | 71.64 | 91.05 | 1.8 min | 4.1x | 6.8 GB | 1.1 min |
| ONNX, standard operators, ONNX Runtime CUDA, no PyTorch (no split, arena kept) | 71.64 | 91.05 | 7.8 min | 1.0x | 37.1 GB (process) |  |
| ONNX, fused fp16 attention (no split, arena kept) | 71.64 | 91.05 | 7.4 min | 1.0x | 27.5 GB (process) |  |
| ONNX, fused, slabs of <= 400k points (1 m context), arena shrunk: the runtime default | 71.55 | 91.06 | 13.0 min | 0.6x | 6.0 GB (process) |  |
| official code today (2025 colour normalization) | 13.74 | 47.46 | 7.5 min | 1.0x | 27.2 GB | 6.8 min |
| official code today, 2023 colour normalization | 22.16 | 59.53 | 7.5 min | 1.0x | 27.2 GB | 6.8 min |

## Curve order at test time

Upstream shuffles the four curves at test time as in training, so inference is random. A
deterministic model needs one fixed order, and since block i uses curve i % 4, the 2-block stages
only ever use the first two. Single pass, Area 5:

| Curve order | mIoU % |
|---|---:|
| z, z-trans, hilbert, hilbert-trans (the checkpoint's order) | 71.02 |
| hilbert, hilbert-trans, z, z-trans | 71.49 |
| hilbert-trans, hilbert, z-trans, z | 71.46 |
| hilbert, z, hilbert-trans, z-trans | 71.38 |
| **z, hilbert, z-trans, hilbert-trans** (`FAST`) | 71.64 |
| random per stage, as upstream (3 seeds) | 71.66 / 71.63 / 71.61 |

Stages that see only one curve family lose 0.2 to 0.6 points. `FAST` alternates the families
(chosen by that rule; the other alternating order scores 0.26 lower, so read differences of a
few tenths here as noise from the choice). The weights don't change, and the order is stored
in the ONNX metadata.

## Parity with the official model

One fragment of each of 5 Area 5 rooms, same weights and seed, against the official code (Dec 2023 release, spconv):

| Backend | Largest logit difference | Points with the same class |
|---|---:|---:|
| math + torch conv | 2.0e-05 | 100.0000 % |
| sdpa + torch conv | 2.0e-05 | 100.0000 % |
| triton + triton conv | 2.8e-05 | 100.0000 % |

## Training step

One optimizer step on a micro-batch of 3 S3DIS training samples (the official recipe's per-GPU batch, 366k points on average): forward, CE + Lovász loss, backward, AdamW. Identical batches for every row.

| Implementation | Precision | ms / step | Speed-up | Peak GPU memory | Points / s |
|---|---:|---:|---:|---:|---:|
| official code | fp16 | 615 | 1.00x | 31.8 GB | 0.59 M |
| this repo, sdpa attention + torch conv | fp16 | 520 | 1.18x | 26.5 GB | 0.70 M |
| this repo, triton attention + triton conv | fp16 | 251 | 2.45x | 10.0 GB | 1.46 M |
| this repo, triton attention + triton conv | bf16 | 253 | 2.43x | 10.0 GB | 1.44 M |

## Inference: time and memory against input size

One forward pass, fp32, real S3DIS fragments (one point per 2 cm voxel). PyTorch memory is the
allocator's peak; ONNX Runtime memory is what its process holds after the run beyond what it held
once the model was loaded (its arena keeps the peak). `onnx-fused` runs attention with ONNX Runtime's
fused fp16 kernel; ONNX times exclude the NumPy preparation of the index tables.

| Points | official | ours-sdpa | ours-triton | onnx | onnx-fused |
|---|---:|---:|---:|---:|---:|
| 50k | 57 ms / 1.21 GB | 56 ms / 0.55 GB | 22 ms / 0.36 GB | 27 ms / 1.12 GB | 26 ms / 0.68 GB |
| 100k | 98 ms / 2.39 GB | 104 ms / 1.10 GB | 27 ms / 0.53 GB | 52 ms / 2.30 GB | 48 ms / 1.58 GB |
| 200k | 185 ms / 4.78 GB | 205 ms / 2.19 GB | 38 ms / 0.85 GB | 108 ms / 4.65 GB | 97 ms / 3.95 GB |
| 400k | 361 ms / 9.56 GB | 408 ms / 4.38 GB | 66 ms / 1.70 GB | 229 ms / 9.35 GB | 208 ms / 7.53 GB |
| 800k | 726 ms / 19.10 GB | 820 ms / 8.75 GB | 121 ms / 3.40 GB | 477 ms / 18.75 GB | 430 ms / 16.98 GB |
| 1600k | 1490 ms / 38.21 GB | 1646 ms / 17.49 GB | 233 ms / 6.80 GB | 997 ms / 37.54 GB | 899 ms / 31.30 GB |

## The two replaced layers

Stage 0 of a training micro-batch (614k points, fp16).

| Layer | Implementation | Pass | ms | Peak GPU memory |
|---|---:|---:|---:|---:|
| rpe-attention stage 0 | upstream | forward | 18.08 | 6.47 GB |
| rpe-attention stage 0 | upstream | forward+backward | 78.83 | 6.54 GB |
| rpe-attention stage 0 | math | forward | 18.57 | 3.81 GB |
| rpe-attention stage 0 | math | forward+backward | 27.73 | 3.81 GB |
| rpe-attention stage 0 | sdpa | forward | 21.98 | 3.81 GB |
| rpe-attention stage 0 | sdpa | forward+backward | 32.80 | 3.83 GB |
| rpe-attention stage 0 | triton | forward | 1.62 | 0.34 GB |
| rpe-attention stage 0 | triton | forward+backward | 13.69 | 0.46 GB |
| subm-conv 3^3 C=32 | spconv | forward | 0.86 | 0.20 GB |
| subm-conv 3^3 C=32 | spconv | forward+backward | 2.11 | 0.31 GB |
| subm-conv 3^3 C=32 | torch (gather-GEMM / pairs) | forward | 10.65 | 0.65 GB |
| subm-conv 3^3 C=32 | torch (gather-GEMM / pairs) | forward+backward | 20.47 | 0.71 GB |
| subm-conv 3^3 C=32 | triton | forward | 0.48 | 0.11 GB |
| subm-conv 3^3 C=32 | triton | forward+backward | 2.68 | 0.18 GB |
| neighbour map 3^3 | sort + searchsorted | build | 2.66 | 0.64 GB |

