from __future__ import annotations

from collections.abc import Iterable

import numpy as np


CORNER_ALIASES = {
    "tl": "top_left",
    "tr": "top_right",
    "bl": "bottom_left",
    "br": "bottom_right",
    "top_left": "top_left",
    "top_right": "top_right",
    "bottom_left": "bottom_left",
    "bottom_right": "bottom_right",
}


def apply_corner_soft_mask(
    image: np.ndarray,
    grid: int = 4,
    corners: Iterable[str] = ("top_left", "top_right", "bottom_right"),
    weight: float = 0.2,
) -> np.ndarray:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("mask 权重必须在 [0, 1] 范围内")
    if image.ndim != 3:
        raise ValueError("image 必须是 HxWxC 格式")
    h, w = image.shape[:2]
    cell_h = max(1, h // grid)
    cell_w = max(1, w // grid)
    original_dtype = image.dtype
    out = image.astype(np.float32, copy=True)
    fill = out.mean(axis=(0, 1), keepdims=True)

    for corner in corners:
        name = CORNER_ALIASES.get(corner)
        if name is None:
            raise ValueError(f"未知的 mask 角落名称: {corner}")
        if name == "top_left":
            y1, y2, x1, x2 = 0, cell_h, 0, cell_w
        elif name == "top_right":
            y1, y2, x1, x2 = 0, cell_h, w - cell_w, w
        elif name == "bottom_left":
            y1, y2, x1, x2 = h - cell_h, h, 0, cell_w
        else:
            y1, y2, x1, x2 = h - cell_h, h, w - cell_w, w
        out[y1:y2, x1:x2] = out[y1:y2, x1:x2] * weight + fill * (1.0 - weight)

    if np.issubdtype(original_dtype, np.integer):
        return np.clip(out, 0, 255).astype(original_dtype)
    return out.astype(original_dtype, copy=False)
