# portable-ptv3

Point Transformer V3 without CUDA extensions. A reimplementation of
[PTv3](https://github.com/Pointcept/PointTransformerV3) (Wu et al., CVPR 2024) that needs no spconv, no
torch_scatter and no flash_attn, adds fused Triton kernels for training, exports to ONNX, and runs
inference with NumPy + ONNX Runtime only. Official Pointcept checkpoints load unchanged.
Tested on Linux with ONNX Runtime's CUDA and CPU providers; the runtime also selects DirectML (AMD and
Intel GPUs on Windows), which ONNX Runtime supports but which has not been tested here.

## Results

Official S3DIS checkpoint, S3DIS Area 5, one GPU (details and every other measurement in
[docs/RESULTS.md](docs/RESULTS.md)):

| | Official code | This repository |
|---|---:|---:|
| mIoU, official protocol (10x test-time augmentation) | 73.53 | 73.47 (fp32), 73.48 (bf16) |
| Evaluation of Area 5, official protocol | 73.7 min, 32.2 GB | 17.0 min (fp32), 13.1 min (bf16), 6.9 / 5.2 GB |
| Training step, official recipe (micro-batch of 3, ~366k points, fp16) | 615 ms, 31.8 GB | 251 ms, 10.0 GB |
| One forward pass, 1.6 M points | 1,490 ms, 38.2 GB | 233 ms, 6.8 GB |
| Inference without PyTorch (ONNX Runtime + NumPy), single pass | n/a | 71.55 mIoU with a 6.0 GB GPU memory peak (PyTorch: 71.64) |
| Same checkpoint, current official code, single pass | 13.74 mIoU | n/a |

- Accuracy: the official training log reports 73.58. With the same seed, every backend here
  matches the official model to within 3e-5 in the logits.
- Speed: it comes from two Triton kernels: a fused RPE attention, and an implicit-GEMM sparse
  convolution with no rulebook.
- Determinism: evaluation here is deterministic. Upstream's is random: it shuffles the
  space-filling curves at test time.
- The last row is the reason for the next section.

## Why

PTv3 is a strong backbone for point cloud segmentation, but the reference code depends on three
CUDA extensions (spconv for sparse convolutions, torch_scatter for pooling, flash_attn for attention)
that must match the CUDA, PyTorch and compiler versions, and don't exist for every platform. That is
fine on a training server and a problem when the model has to run on someone else's laptop.
This repository keeps the model, its weights and its accuracy, and replaces each extension:

| Upstream dependency | Replaced by |
|---|---|
| `spconv.SubMConv3d` | neighbour map from one sort + `searchsorted`; gather-GEMM (inference), per-offset pairs (training), or a Triton implicit-GEMM kernel with an atomic backward pass ([`ptv3/sparse_conv.py`](ptv3/sparse_conv.py), [`ptv3/kernels/subm_conv.py`](ptv3/kernels/subm_conv.py)) |
| `flash_attn` / the unfused RPE attention | `scaled_dot_product_attention`, or a fused Triton kernel that computes the relative position bias in registers, forward and backward ([`ptv3/attention.py`](ptv3/attention.py), [`ptv3/kernels/rpe_attention.py`](ptv3/kernels/rpe_attention.py)) |
| `torch_scatter` | `Tensor.scatter_reduce` |
| PyTorch at inference time | NumPy index preparation + an ONNX graph of standard operators ([`ptv3/export/`](ptv3/export), [`ptv3/runtime/onnx_infer.py`](ptv3/runtime/onnx_infer.py)) |

How each piece works: [docs/DESIGN.md](docs/DESIGN.md). Sizing the ONNX runtime for a machine (points
per run, GPU and RAM budget, ONNX Runtime settings): [docs/DEPLOY.md](docs/DEPLOY.md).

## Quick start

```bash
pip install -e ".[torch,onnx,export,data,dev]"
python scripts/download.py --data data --weights weights          # S3DIS (2 GB) + official checkpoint

# evaluate the official checkpoint with the official protocol (10x test-time augmentation)
python scripts/evaluate.py --checkpoint weights/s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth \
    --data-root data/s3dis --attn-backend triton --conv-backend triton --out results/ours.json

# export to ONNX and run without PyTorch
python scripts/export_onnx.py --checkpoint weights/s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth \
    --out ptv3_s3dis.onnx --fused-attention
python -m ptv3.runtime.onnx_infer --model ptv3_s3dis.onnx --data-root data/s3dis --provider cuda \
    --max-points 400000                                          # ~6 GB of GPU memory
```

Training with the official S3DIS recipe (AdamW, OneCycle, fp16, Mix3D-style mixing, 100 epochs):

```bash
python -m ptv3.train --data-root data/s3dis --out runs/s3dis --attn-backend triton
```

Compare against the official implementation (needs spconv and a clone of the official repository):

```bash
git clone https://github.com/Pointcept/PointTransformerV3 third_party/PointTransformerV3
python scripts/parity_official.py --official-repo third_party/PointTransformerV3 \
    --checkpoint weights/s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth --data-root data/s3dis
```

## Quirks of the official code

Found while matching the official checkpoint bit for bit (details and switches in
[docs/DESIGN.md §6](docs/DESIGN.md#6-behaviour-of-the-official-code-and-the-options-that-reproduce-it)):

- The official checkpoints no longer match the official code. They were trained in December
  2023; since then upstream changed what each block's positional-encoding convolution reads
  (2024-03) and how colours are normalized (2025-03). With today's code the S3DIS checkpoint runs as
  a different network and its accuracy collapses; this repository reproduces the 2023 behaviour.

- Inference is random. The order of the four space-filling curves is shuffled with `randperm` at
  test time too, in the serialization and in every pooling layer, so two runs on the same scene differ.
- A sample's output depends on its batch. Without flash attention the patch size becomes the
  smallest sample of the batch, and the Hilbert curve's depth comes from the batch's largest coordinate.
- The first convolution of each decoder stage never sees the upsampled features. Unpooling updates
  `point.feat` but not the spconv tensor, so that convolution reads the skip connection alone. Official
  checkpoints were trained this way; this repository reproduces it by default.

With the same weights and random seed, every backend here matches the official model (spconv) to
within 3e-5 in the logits and gives the same class for every point of five Area 5 rooms
([docs/RESULTS.md](docs/RESULTS.md)).

## Tests

```bash
pytest                                  # CPU; CUDA tests run when a GPU is present
TRITON_INTERPRET=1 pytest               # also runs the Triton kernels on the CPU (this is what CI does)
PTV3_OFFICIAL_REPO=third_party/PointTransformerV3 pytest tests/test_serialization.py   # against the official encoders
```

The tests compare the sparse convolution with a dictionary-based reference, the patch layout with
upstream's padding loop, the SDPA and Triton attention with the reference math (values and gradients,
including the RPE table's), the serialization with the official encoders, the exported ONNX graph with
the PyTorch model, and the Lovász loss with Pointcept's per-class loop.

## Layout

```
ptv3/
  model.py            PointTransformerV3 + Segmentor (Pointcept parameter names)
  attention.py        patch layout, RPE, SDPA / math / flash / triton backends
  sparse_conv.py      neighbour map, gather-GEMM and pairs convolutions
  serialization.py    z-order and Hilbert codes on integers
  kernels/            Triton: fused RPE attention, implicit-GEMM sparse convolution
  data/               S3DIS loader, Pointcept-equivalent transforms
  evaluate.py         official test protocol (fragments + TTA), batched
  train.py            official S3DIS recipe
  export/             NumPy index preparation + exportable graph
  runtime/            ONNX Runtime inference, no PyTorch
scripts/              download, evaluate, export, parity and benchmark scripts
docs/                 design notes, results
```

## Acknowledgements

PTv3 and Pointcept by Xiaoyang Wu et al. (MIT). S3DIS by Armeni et al. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

```bibtex
@inproceedings{wu2024ptv3,
  title={Point Transformer V3: Simpler, Faster, Stronger},
  author={Wu, Xiaoyang and Jiang, Li and Wang, Peng-Shuai and Liu, Zhijian and Liu, Xihui and Qiao, Yu and Ouyang, Wanli and He, Tong and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```
