import argparse
import json
import shutil
from pathlib import Path

import torch

from model_def import build_model


def load_meta(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text())
    # normalize classes mapping keys to int
    classes = meta.get("classes", {})
    if classes and isinstance(next(iter(classes.keys())), str):
        meta["classes"] = {int(k): v for k, v in classes.items()}
    return meta


def copy_artifacts(model_pt: Path, meta_json: Path, backend_model_dir: Path) -> None:
    backend_model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_pt, backend_model_dir / "model.pt")
    shutil.copy2(meta_json, backend_model_dir / "model_meta.json")
    print(f"[OK] Copied model.pt + model_meta.json to: {backend_model_dir}")


def export_torchscript(model, dummy_input, out_path: Path) -> None:
    model.eval()
    traced = torch.jit.trace(model, dummy_input)
    traced.save(str(out_path))
    print(f"[OK] TorchScript saved: {out_path}")


def export_onnx(model, dummy_input, out_path: Path) -> None:
    model.eval()
    try:
        import onnx  # noqa: F401
    except Exception:
        raise RuntimeError(
            "ONNX export requested, but 'onnx' is not installed. "
            "Install it with: pip install onnx"
        )

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"[OK] ONNX saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Export model artifacts for deployment.")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Training output dir with model.pt + model_meta.json")
    parser.add_argument("--to_backend", type=str, default="../backend/model", help="Where to copy deployment artifacts")

    parser.add_argument("--export_torchscript", action="store_true", help="Also export TorchScript model.ts")
    parser.add_argument("--export_onnx", action="store_true", help="Also export ONNX model.onnx")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    model_pt = out_dir / "model.pt"
    meta_json = out_dir / "model_meta.json"

    if not model_pt.exists():
        raise FileNotFoundError(f"Missing weights file: {model_pt}")
    if not meta_json.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_json}")

    meta = load_meta(meta_json)

    # Always copy artifacts first (MVP deployment path)
    backend_model_dir = Path(args.to_backend)
    copy_artifacts(model_pt, meta_json, backend_model_dir)

    # Optional exports (TorchScript / ONNX)
    if args.export_torchscript or args.export_onnx:
        backbone = meta["backbone"]
        img_size = int(meta["img_size"])
        num_classes = len(meta["classes"])

        device = "cpu"  # export on CPU for portability
        model = build_model(num_classes=num_classes, backbone=backbone, pretrained=False).to(device)
        state = torch.load(model_pt, map_location=device)
        model.load_state_dict(state)

        dummy = torch.randn(1, 3, img_size, img_size, device=device)

        if args.export_torchscript:
            export_torchscript(model, dummy, out_dir / "model.ts")

        if args.export_onnx:
            export_onnx(model, dummy, out_dir / "model.onnx")

    print("[DONE] Export completed successfully.")


if __name__ == "__main__":
    main()
