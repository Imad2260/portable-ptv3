import torch

from conftest import make_batch
from ptv3.checkpoint import build
from ptv3.losses import SegLoss, lovasz_softmax
from ptv3.presets import FAST, OFFICIAL

SMALL = dict(enc_depths=(1, 1, 1, 1, 1), dec_depths=(1, 1, 1, 1))


def small_model(**options):
    torch.manual_seed(0)
    return build("s3dis-rpe", **{**SMALL, **options})


def clone(batch):
    return {k: v.clone() for k, v in batch.items()}


def test_official_checkpoint_names():
    """Parameter names and shapes of the S3DIS preset = the official checkpoint's (46.2 M parameters)."""
    model = build("s3dis-rpe")
    assert sum(p.numel() for p in model.parameters()) == 46_189_708 + 64 * 13 + 13
    names = set(model.state_dict())
    assert "backbone.enc.enc0.block0.cpe.0.weight" in names
    assert "backbone.dec.dec0.block1.attn.rpe.rpe_table" in names
    assert model.state_dict()["backbone.embedding.stem.conv.weight"].shape == (32, 5, 5, 5, 6)


def test_eval_is_deterministic_without_shuffle():
    model = small_model(**FAST).eval()
    batch = make_batch([900, 700])
    with torch.no_grad():
        a, b = model(clone(batch)), model(clone(batch))
    assert torch.equal(a, b)


def test_sdpa_matches_math_on_the_whole_network():
    batch = make_batch([900, 700])
    m1 = small_model(**OFFICIAL).eval()
    m2 = small_model(**{**OFFICIAL, "attn_backend": "sdpa"}).eval()
    torch.manual_seed(1)
    with torch.no_grad():
        a = m1(clone(batch))
    torch.manual_seed(1)
    with torch.no_grad():
        b = m2(clone(batch))
    assert torch.allclose(a, b, atol=1e-4)


def test_train_step_with_duplicate_voxels():
    """Mixed samples (two scenes, one offset) contain voxels that coincide; training must still work."""
    model = small_model(**FAST).train()
    batch = make_batch([800, 800])
    batch["offset"] = batch["offset"][-1:]                      # merge the two scenes into one sample
    labels = torch.randint(0, 13, (len(batch["coord"]),))
    loss = SegLoss()(model(batch), labels)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert torch.isfinite(loss) and all(g is not None and torch.isfinite(g).all() for g in grads)


def lovasz_loop(logits, labels):
    """Pointcept's per-class loop, written out."""
    probs = logits.softmax(1)
    losses = []
    for c in labels.unique():
        fg = (labels == c).float()
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        gt = fg[perm]
        gts = gt.sum()
        jac = 1.0 - (gts - gt.cumsum(0)) / (gts + (1 - gt).cumsum(0))
        jac[1:] = jac[1:] - jac[:-1]
        losses.append(torch.dot(errors_sorted, jac))
    return torch.stack(losses).mean()


def test_lovasz_matches_per_class_loop():
    torch.manual_seed(0)
    logits = torch.randn(5000, 13)
    labels = torch.randint(0, 9, (5000,))
    labels[:100] = -1
    keep = labels != -1
    assert torch.allclose(lovasz_softmax(logits, labels), lovasz_loop(logits[keep], labels[keep]), atol=1e-6)


def test_fast_output_does_not_depend_on_the_batch():
    """With serialization_depth="sample" a sample's Hilbert curve ignores its batch neighbours: here a
    neighbour twice as large, which raises the batch's curve depth by 1 (with the batch's depth the
    logits move by ~0.09; by 3 levels they wouldn't: this Hilbert construction repeats its orientation
    every 3 levels). One block per stage uses only the first curve, so it is a Hilbert curve here."""
    model = small_model(**{**FAST, "order": ("hilbert", "z", "hilbert-trans", "z-trans")}).eval()
    alone = make_batch([900])
    other = make_batch([600], seed=5)
    other["grid_coord"] = other["grid_coord"] * 2
    other["coord"] = other["coord"] * 2
    both = {k: torch.cat([alone[k], other[k]]) for k in ("coord", "grid_coord", "feat")}
    n = len(alone["coord"])
    both["offset"] = torch.tensor([n, n + len(other["coord"])])
    with torch.no_grad():
        a = model(clone(alone))
        b = model(both)[:n]
    assert torch.allclose(a, b, atol=1e-5)
