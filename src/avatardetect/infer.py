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


def softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float32)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / max(float(exp.sum()), 1e-12)


def element_order_from_prototypes(df: pd.DataFrame) -> list[str]:
    return sorted(str(value) for value in df["element_type"].astype(str).unique() if str(value))


def element_probabilities_from_outputs(
    outputs: list[np.ndarray],
    prototypes: pd.DataFrame,
) -> tuple[dict[str, float], str]:
    if len(outputs) < 3:
        return {}, ""
    logits = np.asarray(outputs[2][0], dtype=np.float32)
    element_order = element_order_from_prototypes(prototypes)
    if logits.shape[0] != len(element_order) or not element_order:
        return {}, ""
    probs = softmax(logits)
    by_element = {element: float(probs[idx]) for idx, element in enumerate(element_order)}
    return by_element, element_order[int(np.argmax(probs))]


def rank_with_element_head(
    prototypes: pd.DataFrame,
    scores: np.ndarray,
    top_k: int,
    element_probs: dict[str, float] | None = None,
    predicted_element: str = "",
    element_min_probability: float = 0.35,
) -> list[tuple[pd.Series, float]]:
    limit = max(1, int(top_k))
    predicted_probability = (element_probs or {}).get(predicted_element, 0.0)
    if not element_probs or not predicted_element or predicted_probability < float(element_min_probability):
        order = np.argsort(-scores)[:limit]
        return [(prototypes.iloc[int(idx)], float(scores[int(idx)])) for idx in order]

    best_by_appearance: dict[str, tuple[int, float]] = {}
    indices_by_appearance_element: dict[tuple[str, str], list[int]] = {}
    for idx, row in prototypes.iterrows():
        appearance_id = appearance_id_from_row(row)
        element_type = str(row.get("element_type", ""))
        score = float(scores[int(idx)])
        best = best_by_appearance.get(appearance_id)
        if best is None or score > best[1]:
            best_by_appearance[appearance_id] = (int(idx), score)
        indices_by_appearance_element.setdefault((appearance_id, element_type), []).append(int(idx))

    ranked_appearances = sorted(best_by_appearance.items(), key=lambda item: item[1][1], reverse=True)
    rows: list[tuple[pd.Series, float]] = []
    for appearance_id, (best_idx, appearance_score) in ranked_appearances:
        candidate_indices = indices_by_appearance_element.get((appearance_id, predicted_element), [])
        if candidate_indices:
            selected_idx = max(candidate_indices, key=lambda idx: float(scores[idx]))
        else:
            selected_idx = best_idx
        rows.append((prototypes.iloc[selected_idx], appearance_score))
        if len(rows) >= limit:
            break
    return rows


def image_to_tensor(image: np.ndarray, cfg: dict) -> np.ndarray:
    tensor = image.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)
    norm = cfg.get("normalization", {})
    mean = np.asarray(norm.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(norm.get("std", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
    return ((tensor - mean) / std)[None, ...].astype(np.float32)


def preprocess_rgb_array(image_rgb: np.ndarray, cfg: dict) -> np.ndarray:
    image_size = cfg["data"].get("image_size", [115, 115])
    width, height = int(image_size[0]), int(image_size[1])
    image = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)
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
    return image


def crop_element_roi(image: np.ndarray, cfg: dict) -> np.ndarray:
    roi_cfg = cfg.get("element_roi", {})
    x = int(roi_cfg.get("x", 0))
    y = int(roi_cfg.get("y", 0))
    size = int(roi_cfg.get("size", 48))
    output_size = roi_cfg.get("output_size", [64, 64])
    output_w = int(output_size[0]) if len(output_size) > 0 else 64
    output_h = int(output_size[1]) if len(output_size) > 1 else output_w

    h, w = image.shape[:2]
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    src_x1 = max(0, x)
    src_y1 = max(0, y)
    src_x2 = min(w, x + size)
    src_y2 = min(h, y + size)
    if src_x1 < src_x2 and src_y1 < src_y2:
        dst_x1 = src_x1 - x
        dst_y1 = src_y1 - y
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    interpolation = cv2.INTER_LINEAR if output_w > size or output_h > size else cv2.INTER_AREA
    return cv2.resize(canvas, (output_w, output_h), interpolation=interpolation)


def preprocess_image(path: str | Path, cfg: dict) -> np.ndarray:
    image = preprocess_rgb_array(np.asarray(Image.open(path).convert("RGB")), cfg)
    return image_to_tensor(image, cfg)


def preprocess_element_image(path: str | Path, cfg: dict) -> np.ndarray:
    image = preprocess_rgb_array(np.asarray(Image.open(path).convert("RGB")), cfg)
    return image_to_tensor(crop_element_roi(image, cfg), cfg)


def preprocess_inputs(path: str | Path, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    image = preprocess_rgb_array(np.asarray(Image.open(path).convert("RGB")), cfg)
    return image_to_tensor(image, cfg), image_to_tensor(crop_element_roi(image, cfg), cfg)


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
    inputs = session.get_inputs()
    image_tensor, element_tensor = preprocess_inputs(image_path, cfg)
    feed = {inputs[0].name: image_tensor}
    if len(inputs) > 1:
        feed[inputs[1].name] = element_tensor
    outputs = session.run(None, feed)
    embedding = outputs[0][0].astype(np.float32)
    embedding = embedding / max(np.linalg.norm(embedding), 1e-12)

    df = pd.read_csv(prototypes_path, dtype=str).fillna("")
    matrix = np.stack([decode_vector(x) for x in df["embedding"]])
    scores = matrix @ embedding
    element_probs, predicted_element = element_probabilities_from_outputs(outputs, df)
    element_min_probability = float(cfg.get("inference", {}).get("element_min_probability", 0.35))
    rows = rank_with_element_head(df, scores, top_k, element_probs, predicted_element, element_min_probability)
    if predicted_element:
        use_text = "已用于重排" if element_probs[predicted_element] >= element_min_probability else "低于阈值，未用于重排"
        print(
            f"元素头预测={predicted_element} 概率={element_probs[predicted_element]:.4f} "
            f"阈值={element_min_probability:.4f} {use_text}"
        )
    for rank, (row, score) in enumerate(rows, start=1):
        skin_name = row.get("skin_name", "")
        appearance_id = appearance_id_from_row(row)
        variant_id = row.get("variant_id", appearance_id)
        element_type = row.get("element_type", "")
        weapon_type = row.get("weapon_type", "")
        element_prob = element_probs.get(element_type)
        element_prob_text = f" 元素概率={element_prob:.4f}" if element_prob is not None else ""
        print(
            f"{rank}: 分数={score:.4f}{element_prob_text} "
            f"character_id={row['character_id']} appearance_id={appearance_id} variant_id={variant_id} "
            f"名称={row['character_name']} skin_id={row['skin_id']} 皮肤={skin_name} "
            f"元素={element_type} element_type={element_type} 武器={weapon_type} weapon_type={weapon_type}"
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
