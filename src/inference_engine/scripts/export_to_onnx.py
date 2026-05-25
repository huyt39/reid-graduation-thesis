"""Export inference_engine PyTorch models to ONNX for Triton serving.

Run from src/inference_engine/ with:
    uv run --with onnx --with onnxruntime python scripts/export_to_onnx.py

Outputs:
    <repo_root>/triton_models/osnet/1/model.onnx
    <repo_root>/triton_models/multi_attr/1/model.onnx   (if weights present)

The exported ONNX models expect:
    input:  float32 [batch, 3, 256, 128]  (RGB, ImageNet-normalised, NCHW)

Preprocessing + L2-normalisation stay client-side in inference_engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_ROOT = Path(__file__).resolve().parents[1]
TRITON_REPO = REPO_ROOT / "triton_models"

sys.path.insert(0, str(INFERENCE_ROOT))


def _verify_onnx(onnx_path: Path, dummy: torch.Tensor, torch_out: torch.Tensor) -> None:
    """Cosine similarity > 0.999 between PyTorch and onnxruntime outputs."""
    import onnxruntime as ort  # type: ignore

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy.numpy().astype(np.float32)})[0]

    t = torch_out.detach().cpu().numpy().reshape(torch_out.shape[0], -1)
    o = onnx_out.reshape(onnx_out.shape[0], -1)
    # cosine sim per row
    t_n = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-8)
    o_n = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-8)
    sim = (t_n * o_n).sum(axis=1).mean()
    print(f"  parity check: cosine sim = {sim:.6f}")
    assert sim > 0.999, f"ONNX export diverges from PyTorch: cos sim {sim}"


def export_osnet() -> None:
    weights = INFERENCE_ROOT / "src" / "assets" / "models" / "osnet" / "model.pth.tar-150"
    if not weights.exists():
        print(f"[skip] osnet weights not found: {weights}")
        return

    print(f"[osnet] loading weights from {weights}")
    from src.models.osnet import osnet_x1_0

    model = osnet_x1_0(num_classes=1000, weight_path=str(weights), device=torch.device("cpu"))
    model.eval()

    out_dir = TRITON_REPO / "osnet" / "1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.onnx"

    dummy = torch.randn(1, 3, 256, 128)
    with torch.no_grad():
        torch_out = model(dummy)

    print(f"[osnet] torch output shape: {tuple(torch_out.shape)}  → exporting")
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[osnet] wrote {out_path} ({out_path.stat().st_size // 1024} KB)")

    # verify with a small batch too — catches dynamic-axis regressions
    _verify_onnx(out_path, dummy, torch_out)
    batch_dummy = torch.randn(4, 3, 256, 128)
    with torch.no_grad():
        batch_torch = model(batch_dummy)
    _verify_onnx(out_path, batch_dummy, batch_torch)


def export_multi_attr() -> None:
    weights = INFERENCE_ROOT / "src" / "assets" / "models" / "multi_attr" / "best_model_multi_attr_b0.pth"
    if not weights.exists():
        print(f"[skip] multi_attr weights not found: {weights}")
        return

    print(f"[multi_attr] loading weights from {weights}")
    try:
        from src.models.multi_attr_classifier import MultiAttrEfficientNetB0
    except Exception as exc:
        print(f"[skip] multi_attr import failed: {exc}")
        return

    wrapper = MultiAttrEfficientNetB0(weight_path=str(weights), device=torch.device("cpu"))
    inner = getattr(wrapper, "model", wrapper)
    inner.eval()

    out_dir = TRITON_REPO / "multi_attr" / "1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.onnx"

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch_out = inner(dummy)
    if isinstance(torch_out, dict):
        output_names = sorted(torch_out.keys())
        dynamic_axes = {"input": {0: "batch"}, **{n: {0: "batch"} for n in output_names}}
    elif isinstance(torch_out, (list, tuple)):
        output_names = [f"out_{i}" for i in range(len(torch_out))]
        dynamic_axes = {"input": {0: "batch"}, **{n: {0: "batch"} for n in output_names}}
    else:
        output_names = ["output"]
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    print(f"[multi_attr] outputs: {output_names}  → exporting")
    torch.onnx.export(
        inner,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[multi_attr] wrote {out_path} ({out_path.stat().st_size // 1024} KB)")


def main() -> None:
    print(f"repo root: {REPO_ROOT}")
    print(f"inference root: {INFERENCE_ROOT}")
    print(f"triton repo: {TRITON_REPO}\n")
    export_osnet()
    print()
    export_multi_attr()


if __name__ == "__main__":
    main()
