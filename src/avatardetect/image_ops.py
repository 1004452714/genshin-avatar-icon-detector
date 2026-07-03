from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def read_rgba(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"))


def read_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def resize_cover(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    h, w = image.shape[:2]
    scale = max(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x1 = max(0, (new_w - target_w) // 2)
    y1 = max(0, (new_h - target_h) // 2)
    return resized[y1 : y1 + target_h, x1 : x1 + target_w]


def resize_rgba(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    interpolation = cv2.INTER_AREA
    if target_w > image.shape[1] or target_h > image.shape[0]:
        interpolation = cv2.INTER_LINEAR
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def overlay_rgba(base: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:
    out = base.copy()
    h, w = out.shape[:2]
    oh, ow = overlay.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + ow)
    y2 = min(h, y + oh)
    if x1 >= x2 or y1 >= y2:
        return out

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    src = overlay[oy1:oy2, ox1:ox2].astype(np.float32) / 255.0
    dst = out[y1:y2, x1:x2].astype(np.float32) / 255.0
    src_alpha = src[..., 3:4]
    dst_alpha = dst[..., 3:4]
    out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
    out_rgb = (src[..., :3] * src_alpha + dst[..., :3] * dst_alpha * (1.0 - src_alpha)) / np.maximum(
        out_alpha,
        1e-6,
    )
    merged = np.concatenate([out_rgb, out_alpha], axis=2)
    out[y1:y2, x1:x2] = np.clip(merged * 255.0, 0, 255).astype(np.uint8)
    return out


def resize_contain_rgba(
    image: np.ndarray,
    size: tuple[int, int],
    scale: float,
    shift_x: int,
    shift_y: int,
) -> np.ndarray:
    target_w, target_h = size
    h, w = image.shape[:2]
    base_scale = min(target_w / w, target_h / h) * scale
    new_w = max(1, int(round(w * base_scale)))
    new_h = max(1, int(round(h * base_scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    x1 = (target_w - new_w) // 2 + shift_x
    y1 = (target_h - new_h) // 2 + shift_y
    x2 = x1 + new_w
    y2 = y1 + new_h

    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(target_w, x2)
    dst_y2 = min(target_h, y2)
    if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
        return canvas
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]
    return canvas


def alpha_blend(foreground_rgba: np.ndarray, background_rgb: np.ndarray) -> np.ndarray:
    fg = foreground_rgba[..., :3].astype(np.float32)
    alpha = foreground_rgba[..., 3:4].astype(np.float32) / 255.0
    bg = background_rgb.astype(np.float32)
    out = fg * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def random_color_drift(image: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    out = image.astype(np.float32)
    brightness = float(cfg.get("brightness", 0.0))
    contrast = float(cfg.get("contrast", 0.0))
    saturation = float(cfg.get("saturation", 0.0))
    hue_shift = float(cfg.get("hue_shift", 0.0))
    gamma_range = cfg.get("gamma", [1.0, 1.0])

    if contrast > 0:
        factor = 1.0 + rng.uniform(-contrast, contrast)
        mean = out.mean(axis=(0, 1), keepdims=True)
        out = (out - mean) * factor + mean
    if brightness > 0:
        factor = 1.0 + rng.uniform(-brightness, brightness)
        out = out * factor

    out = np.clip(out, 0, 255).astype(np.uint8)
    if saturation > 0 or hue_shift > 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        if hue_shift > 0:
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-hue_shift, hue_shift) * 180.0) % 180.0
        if saturation > 0:
            hsv[..., 1] *= 1.0 + rng.uniform(-saturation, saturation)
        hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if len(gamma_range) == 2:
        gamma = float(rng.uniform(gamma_range[0], gamma_range[1]))
        if gamma > 0:
            table = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255).astype(np.uint8)
            out = cv2.LUT(out, table)
    return out


def random_degrade(image: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    out = image.copy()
    noise_std = float(cfg.get("noise_std", 0.0))
    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < float(cfg.get("blur_probability", 0.0)):
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if rng.random() < float(cfg.get("sharpen_probability", 0.0)):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(out, -1, kernel)
    if rng.random() < float(cfg.get("jpeg_probability", 0.0)):
        quality = int(rng.integers(78, 96))
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out


def solid_background(size: tuple[int, int], rarity: Any) -> np.ndarray:
    colors = {
        "4": (146, 92, 190),
        "5": (190, 128, 58),
    }
    color = colors.get(str(rarity), (120, 100, 80))
    w, h = size
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    bg[:, :] = color
    return bg
