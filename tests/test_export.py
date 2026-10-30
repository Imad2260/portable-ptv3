import numpy as np
import pytest
import torch

from conftest import make_batch
from ptv3.checkpoint import build
from ptv3.export.graph import ExportGraph
from ptv3.export.prep import collate, prepare
from ptv3.presets import FAST, S3DIS_RPE

SMALL = dict(enc_depths=(1, 1, 1, 2, 1), dec_depths=(1, 1, 1, 1))


@pytest.mark.parametrize("cpe_input", ["mlp_branch", "residual"])
def test_graph_on_prepared_tables_equals_model(cpe_input):
    torch.manual_seed(0)
    model = build("s3dis-rpe", **{**SMALL, **FAST, "cpe_input": cpe_input}).eval()
    for m in model.modules():                       # non-trivial BatchNorm statistics
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean.normal_()
            m.running_var.uniform_(0.5, 2)
    batch = make_batch([1500, 1500], seed=3)
    cfg = {**S3DIS_RPE, **SMALL, **FAST, "cpe_input": cpe_input}
    samples, start = [], 0
    for end in batch["offset"].tolist():
        samples.append(prepare(batch["grid_coord"][start:end].numpy(), batch["feat"][start:end].numpy(), cfg))
        start = end
    tables = collate(samples, cfg)
    graph = ExportGraph(model, list(tables))
    with torch.no_grad():
        want = model({k: v.clone() for k, v in batch.items()})
        got = graph(*[torch.from_numpy(v) for v in tables.values()])
    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()


def test_short_sample_uses_masked_patch():
    torch.manual_seed(0)
    model = build("s3dis-rpe", **{**SMALL, **FAST, "patch_mode": "fixed"}).eval()
    batch = make_batch([90], seed=5)                 # fewer points than a patch
    tables = prepare(batch["grid_coord"].numpy(), batch["feat"].numpy(), {**S3DIS_RPE, **SMALL, **FAST, "patch_mode": "fixed"})
    tables = collate([tables], {**S3DIS_RPE, **SMALL, **FAST, "patch_mode": "fixed"})
    assert any(k.endswith("_mask") for k in tables)
    graph = ExportGraph(model, list(tables))
    with torch.no_grad():
        want = model({k: v.clone() for k, v in batch.items()})
        got = graph(*[torch.from_numpy(v) for v in tables.values()])
    assert torch.allclose(got, want, atol=1e-4)


@pytest.mark.parametrize("fused", [False, True])
def test_onnx_runtime_roundtrip(tmp_path, fused):
    """Export a small model, run it in ONNX Runtime (CPU), compare with PyTorch."""
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    from torch.export import Dim

    torch.manual_seed(0)
    cfg = {**S3DIS_RPE, **SMALL, **FAST}
    model = build("s3dis-rpe", **{**SMALL, **FAST}).eval()

    def tables_for(sizes, seed):
        batch = make_batch(sizes, seed=seed)
        samples, start = [], 0
        for end in batch["offset"].tolist():
            samples.append(prepare(batch["grid_coord"][start:end].numpy(), batch["feat"][start:end].numpy(), cfg))
            start = end
        return batch, collate(samples, cfg)

    _, tables = tables_for([2000, 1200], 7)
    names = list(tables)
    fixed = ("feat0", "_nbr", "_children", "_grid")
    dynamic = (tuple({d: Dim.DYNAMIC for d in range(v.ndim - (1 if n.endswith(fixed) else 0))} for n, v in tables.items()),)
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("export_onnx", os.path.join(os.path.dirname(__file__), "..", "scripts", "export_onnx.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with torch.no_grad():
        table = mod.multi_head_attention_translation() if fused else None
        program = torch.onnx.export(ExportGraph(model, names, fused_attention=fused), tuple(torch.from_numpy(v) for v in tables.values()),
                                    dynamo=True, opset_version=18, input_names=names, output_names=["logits"],
                                    dynamic_shapes=dynamic, custom_translation_table=table)
    path = str(tmp_path / "m.onnx")
    program.save(path)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    batch, tables = tables_for([1700, 900, 400], 11)          # other sizes than the traced ones
    got = session.run(None, {n: tables[n] for n in names})[0]
    with torch.no_grad():
        want = model({k: v.clone() for k, v in batch.items()}).numpy()
    assert np.abs(got - want).max() < (2e-2 if fused else 1e-3)        # fused: fp16 attention
