from __future__ import annotations

import argparse
import base64
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

from .config import load_config
from .mask import apply_corner_soft_mask


def decode_vector(text: str) -> np.ndarray:
    vec = np.frombuffer(base64.b64decode(text), dtype="<f4").astype(np.float32)
    return vec / max(np.linalg.norm(vec), 1e-12)


def appearance_id_from_row(row: pd.Series) -> str:
    value = str(row.get("appearance_id", "") or "")
    if value:
        return value
    return f"{row['character_id']}_{row['skin_id']}"


def preprocess_image(path: str | Path, cfg: dict) -> np.ndarray:
    image_size = cfg["data"].get("image_size", [115, 115])
    width, height = int(image_size[0]), int(image_size[1])
    image = np.asarray(Image.open(path).convert("RGB"))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    mask_cfg = cfg.get("mask", {})
    if mask_cfg.get("enabled", True):
        image = apply_corner_soft_mask(
            image,
            grid=int(mask_cfg.get("grid", 4)),
            corners=mask_cfg.get(
                "inference_corners",
                mask_cfg.get("corners", ["top_left", "top_right", "bottom_right"]),
            ),
            weight=float(mask_cfg.get("inference_weight", 0.2)),
        )
    tensor = image.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)
    norm = cfg.get("normalization", {})
    mean = np.asarray(norm.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(norm.get("std", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor[None, ...].astype(np.float32)


def infer(
    config_path: str | Path,
    model_path: str | Path,
    prototypes_path: str | Path,
    image_path: str | Path,
    top_k: int,
    provider: str,
) -> None:
    cfg = load_config(config_path)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    embedding = session.run(None, {input_name: preprocess_image(image_path, cfg)})[0][0].astype(np.float32)
    embedding = embedding / max(np.linalg.norm(embedding), 1e-12)

    df = pd.read_csv(prototypes_path, dtype=str).fillna("")
    matrix = np.stack([decode_vector(x) for x in df["embedding"]])
    scores = matrix @ embedding
    order = np.argsort(-scores)[:top_k]
    for rank, idx in enumerate(order, start=1):
        row = df.iloc[int(idx)]
        skin_name = row.get("skin_name", "")
        appearance_id = appearance_id_from_row(row)
        variant_id = row.get("variant_id", appearance_id)
        element_type = row.get("element_type", "")
        print(
            f"{rank}: 分数={scores[idx]:.4f} "
            f"character_id={row['character_id']} appearance_id={appearance_id} variant_id={variant_id} "
            f"名称={row['character_name']} skin_id={row['skin_id']} 皮肤={skin_name} "
            f"元素={element_type} element_type={element_type}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prototypes", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(args.config, args.model, args.prototypes, args.image, args.top_k, args.provider)
