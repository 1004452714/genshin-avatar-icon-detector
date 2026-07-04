from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoint import load_model_from_checkpoint
from .config import load_config, project_path
from .model import OnnxAvatarWrapper


def export_onnx(config_path: str | Path, checkpoint_path: str | Path, out_path: str | Path | None) -> None:
    cfg = load_config(config_path)
    device = torch.device("cpu")
    model, _ = load_model_from_checkpoint(checkpoint_path, device)
    wrapper = OnnxAvatarWrapper(model).eval()
    image_size = cfg["data"].get("image_size", [115, 115])
    width, height = int(image_size[0]), int(image_size[1])
    dummy = torch.randn(1, 3, height, width, dtype=torch.float32)
    roi_size = cfg.get("element_roi", {}).get("output_size", [64, 64])
    roi_width = int(roi_size[0]) if len(roi_size) > 0 else 64
    roi_height = int(roi_size[1]) if len(roi_size) > 1 else roi_width
    dummy_element = torch.randn(1, 3, roi_height, roi_width, dtype=torch.float32)
    output = Path(out_path) if out_path else project_path(cfg, cfg["export"]["onnx_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy, dummy_element),
        output,
        input_names=["input_image", "input_element_image"],
        output_names=["embedding", "rarity_logits", "element_logits"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "input_element_image": {0: "batch_size"},
            "embedding": {0: "batch_size"},
            "rarity_logits": {0: "batch_size"},
            "element_logits": {0: "batch_size"},
        },
        opset_version=int(cfg.get("export", {}).get("opset", 17)),
    )
    print(f"已写入: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_onnx(args.config, args.checkpoint, args.out)
