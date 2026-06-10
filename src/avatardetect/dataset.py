from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .image_ops import (
    alpha_blend,
    random_color_drift,
    random_degrade,
    read_rgb,
    read_rgba,
    resize_contain_rgba,
    resize_cover,
    solid_background,
)
from .mask import apply_corner_soft_mask


REQUIRED_COLUMNS = {
    "appearance_id",
    "character_id",
    "character_name",
    "skin_id",
    "rarity",
    "image_path",
    "background_path",
}


def load_labels(path: str | Path) -> pd.DataFrame:
    labels_path = Path(path)
    if not labels_path.exists():
        raise FileNotFoundError(f"找不到 labels csv: {labels_path}")
    df = pd.read_csv(labels_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"labels csv 缺少字段: {sorted(missing)}")
    if df.empty:
        raise ValueError("labels csv 是空文件")
    return df.fillna("")


def make_mappings(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    appearances = sorted(df["appearance_id"].astype(str).unique())
    rarities = sorted(str(x) for x in df["rarity"].astype(str).unique() if str(x))
    return {
        "appearance_to_idx": {name: i for i, name in enumerate(appearances)},
        "idx_to_appearance": {str(i): name for i, name in enumerate(appearances)},
        "rarity_to_idx": {name: i for i, name in enumerate(rarities)},
        "idx_to_rarity": {str(i): name for i, name in enumerate(rarities)},
    }


def split_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "split" in df.columns and df["split"].astype(str).str.len().gt(0).any():
        split = df["split"].astype(str).str.lower()
        train_df = df[split.eq("train")].copy()
        val_df = df[split.isin(["val", "valid", "validation"])].copy()
        if train_df.empty:
            raise ValueError("labels csv 有 split 字段，但没有 train 行")
        if val_df.empty:
            val_df = train_df.copy()
        return train_df, val_df
    return df.copy(), df.copy()


def resolve_data_path(root: Path, value: Any) -> Path | None:
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return root / path


class AvatarDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root: str | Path,
        cfg: dict[str, Any],
        mappings: dict[str, dict[str, int]],
        train: bool,
        seed: int = 42,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.cfg = cfg
        self.train = train
        self.mappings = mappings
        self.rng = np.random.default_rng(seed)
        image_size = cfg["data"].get("image_size", [115, 115])
        self.size = (int(image_size[0]), int(image_size[1]))
        norm = cfg.get("normalization", {})
        self.mean = np.asarray(norm.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(norm.get("std", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image = self.render_row(row)
        tensor = image.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)
        tensor = (tensor - self.mean) / self.std

        appearance_id = str(row["appearance_id"])
        rarity = str(row["rarity"])
        rarity_idx = self.mappings["rarity_to_idx"].get(rarity, -1)
        return {
            "image": torch.from_numpy(tensor).float(),
            "appearance_idx": torch.tensor(self.mappings["appearance_to_idx"][appearance_id], dtype=torch.long),
            "rarity_idx": torch.tensor(rarity_idx, dtype=torch.long),
            "appearance_id": appearance_id,
            "character_id": str(row["character_id"]),
        }

    def render_row(self, row: pd.Series) -> np.ndarray:
        augment_cfg = self.cfg.get("augment", {})
        avatar_path = resolve_data_path(self.root, row["image_path"])
        if avatar_path is None or not avatar_path.exists():
            raise FileNotFoundError(f"找不到角色图: {avatar_path}")

        bg_path = resolve_data_path(self.root, row["background_path"])
        if bg_path and bg_path.exists():
            background = resize_cover(read_rgb(bg_path), self.size)
        else:
            background = solid_background(self.size, row.get("rarity", ""))

        avatar = read_rgba(avatar_path)
        if self.train:
            scale = float(self.rng.uniform(*augment_cfg.get("scale_range", [1.0, 1.0])))
            translate = int(augment_cfg.get("translate_px", 0))
            shift_x = int(self.rng.integers(-translate, translate + 1)) if translate > 0 else 0
            shift_y = int(self.rng.integers(-translate, translate + 1)) if translate > 0 else 0
        else:
            scale = 1.0
            shift_x = 0
            shift_y = 0

        foreground = resize_contain_rgba(avatar, self.size, scale, shift_x, shift_y)
        image = alpha_blend(foreground, background)
        if self.train:
            image = random_color_drift(image, augment_cfg, self.rng)
            image = random_degrade(image, augment_cfg, self.rng)

        mask_cfg = self.cfg.get("mask", {})
        if mask_cfg.get("enabled", True):
            if self.train:
                weight_range = mask_cfg.get("train_weight_range", [0.2, 0.2])
                weight = float(self.rng.uniform(weight_range[0], weight_range[1]))
            else:
                weight = float(mask_cfg.get("inference_weight", 0.2))
            image = apply_corner_soft_mask(
                image,
                grid=int(mask_cfg.get("grid", 4)),
                corners=mask_cfg.get("corners", ["top_left", "top_right", "bottom_right"]),
                weight=weight,
            )
        return image
