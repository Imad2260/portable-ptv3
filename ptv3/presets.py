"""Model settings of the official PTv3 checkpoints (Pointcept/PointTransformerV3 on the Hugging Face Hub).

`UPSTREAM` reproduces the current official code exactly (for parity tests). `OFFICIAL` reproduces the
code the official checkpoints were trained with. `FAST` is `OFFICIAL` with deterministic evaluation and
SDPA attention: same weights, nothing that needs retraining.
"""
S3DIS_RPE = dict(
    num_classes=13,
    backbone_out_channels=64,
    in_channels=6,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(128, 128, 128, 128, 128),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(128, 128, 128, 128),
    mlp_ratio=4,
    qkv_bias=True,
    drop_path=0.3,
    shuffle_orders=True,
    enable_rpe=True,
    enable_flash=False,
    upcast_attention=True,
    upcast_softmax=True,
)

# current upstream code (github.com/Pointcept/PointTransformerV3 and Pointcept since 2024-03-26)
UPSTREAM = dict(attn_backend="math", patch_mode="min", serialization_depth=None,
                unpool_conv_input="skip", cpe_input="residual", shuffle_at_eval=True)
# the code the official checkpoints were trained with (Pointcept, Dec 2023): each block's CPE convolution
# reads the previous block's MLP branch output. Current upstream code runs these checkpoints as a
# different network, and their accuracy collapses (docs/RESULTS.md).
OFFICIAL = dict(UPSTREAM, cpe_input="mlp_branch")
# Deterministic evaluation needs one fixed curve order. Blocks use order i % 4, so in the 2-block stages
# only the first two curves are ever used: with the checkpoint's order (z, z-trans, ...) no Hilbert
# curve outside stage 3. Training shuffled the orders, so every stage saw both families; alternating
# them keeps that. Single-pass Area 5 mIoU: z,z-trans,... 71.02; z,hilbert,... 71.64; random shuffles
# as upstream 71.61-71.66 (docs/RESULTS.md). Only the order of the curves changes, not the weights.
FAST_ORDER = ("z", "hilbert", "z-trans", "hilbert-trans")
# serialization_depth="sample": each sample's Hilbert curve at its own depth, as when upstream runs
# one fragment per pass (the official test protocol), so batching fragments changes nothing.
FAST = dict(OFFICIAL, attn_backend="sdpa", shuffle_at_eval=False, order=FAST_ORDER, serialization_depth="sample")

PRESETS = {"s3dis-rpe": S3DIS_RPE}
HF_REPO = "Pointcept/PointTransformerV3"
HF_FILES = {"s3dis-rpe": "s3dis-semseg-pt-v3m1-0-rpe/model/model_best.pth"}
