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
    overlay_rgba,
    resize_rgba,
    resize_contain_rgba,
    resize_cover,
    solid_background,
)
from .mask import apply_corner_soft_mask


REQUIRED_COLUMNS = {
    "variant_id",
    "character_id",
    "character_name",
    "skin_id",
    "skin_name",
    "element_type",
    "rarity",
    "image_path",
    "element_icon_path",
    "background_path",
}

ALLOWED_ELEMENT_TYPES = {"冰", "风", "雷", "水", "火", "岩", "草"}


def load_labels(path: str | Path) -> pd.DataFrame:
    labels_path = Path(path)
    if not labels_path.exists():
        raise FileNotFoundError(f"找不到 labels csv: {labels_path}")
    df = pd.read_csv(labels_path, dtype=str).fillna("")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"labels csv 缺少字段: {sorted(missing)}")
    if df.empty:
        raise ValueError("labels csv 是空文件")
    derived_appearance = df["character_id"].astype(str) + "_" + df["skin_id"].astype(str)
    if "appearance_id" not in df.columns:
        df["appearance_id"] = derived_appearance
    else:
        df["appearance_id"] = df["appearance_id"].where(df["appearance_id"].astype(str).str.len().gt(0), derived_appearance)
    return df


def make_mappings(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    variants = sorted(df["variant_id"].astype(str).unique())
    appearances = sorted(df["appearance_id"].astype(str).unique())
    rarities = sorted(str(x) for x in df["rarity"].astype(str).unique() if str(x))
    return {
        "variant_to_idx": {name: i for i, name in enumerate(variants)},
        "idx_to_variant": {str(i): name for i, name in enumerate(variants)},
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
        views_per_sample: int = 1,
        prototype: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.cfg = cfg
        self.train = train
        self.prototype = prototype
        self.mappings = mappings
        self.rng = np.random.default_rng(seed)
        self.views_per_sample = max(1, int(views_per_sample))
        image_size = cfg["data"].get("image_size", [115, 115])
        self.size = (int(image_size[0]), int(image_size[1]))
        norm = cfg.get("normalization", {})
        self.mean = np.asarray(norm.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(norm.get("std", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        if self.train and self.views_per_sample > 1:
            tensors = [self.image_to_tensor(self.render_row(row)) for _ in range(self.views_per_sample)]
            tensor = np.stack(tensors, axis=0)
        else:
            tensor = self.image_to_tensor(self.render_row(row))

        variant_id = str(row["variant_id"])
        appearance_id = str(row["appearance_id"])
        rarity = str(row["rarity"])
        rarity_idx = self.mappings["rarity_to_idx"].get(rarity, -1)
        variant_idx = self.mappings["variant_to_idx"][variant_id]
        return {
            "image": torch.from_numpy(tensor).float(),
            "variant_idx": torch.tensor(variant_idx, dtype=torch.long),
            "appearance_idx": torch.tensor(variant_idx, dtype=torch.long),
            "rarity_idx": torch.tensor(rarity_idx, dtype=torch.long),
            "variant_id": variant_id,
            "appearance_id": appearance_id,
            "character_id": str(row["character_id"]),
        }

    def image_to_tensor(self, image: np.ndarray) -> np.ndarray:
        tensor = image.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)
        return (tensor - self.mean) / self.std

    def jitter(self, max_abs_px: int) -> int:
        if max_abs_px <= 0 or not self.train or self.prototype:
            return 0
        return int(self.rng.integers(-max_abs_px, max_abs_px + 1))

    def compose_overlays(self, avatar: np.ndarray, row: pd.Series) -> np.ndarray:
        compose_cfg = self.cfg.get("compose", {})
        element_path = resolve_data_path(self.root, row["element_icon_path"])
        if element_path is None or not element_path.exists():
            raise FileNotFoundError(f"找不到元素图标: {element_path}")

        element_icon = read_rgba(element_path)
        element_offset = compose_cfg.get("element_offset", [7, 7])
        element_x = int(element_offset[0]) if len(element_offset) > 0 else 7
        element_y = int(element_offset[1]) if len(element_offset) > 1 else 7
        element_jitter_px = int(compose_cfg.get("element_jitter_px", 0))
        out = overlay_rgba(
            avatar,
            element_icon,
            element_x + self.jitter(element_jitter_px),
            element_y + self.jitter(element_jitter_px),
        )

        training_cfg = compose_cfg.get("training_icon", {})
        if not training_cfg.get("enabled", True):
            return out

        icon_paths = training_cfg.get(
            "paths",
            [
                "",
                "assets/overlays/training/UI_TrainingGuide_Promote.png",
                "assets/overlays/training/UI_TrainingGuide_Finish.png",
            ],
        )
        icon_path_text = ""
        if self.train:
            icon_path_text = str(self.rng.choice(icon_paths))
        else:
            icon_path_text = str(training_cfg.get("inference_path", ""))
        if not icon_path_text:
            return out

        icon_path = resolve_data_path(self.root, icon_path_text)
        if icon_path is None or not icon_path.exists():
            raise FileNotFoundError(f"找不到养成图标: {icon_path}")
        training_icon = read_rgba(icon_path)
        if self.train and not self.prototype:
            scale_range = training_cfg.get("scale_range", [1.5, 1.5])
            training_scale = float(self.rng.uniform(float(scale_range[0]), float(scale_range[1])))
        else:
            training_scale = float(training_cfg.get("scale", 1.5))
        new_w = max(1, int(round(training_icon.shape[1] * training_scale)))
        new_h = max(1, int(round(training_icon.shape[0] * training_scale)))
        training_icon = resize_rgba(training_icon, (new_w, new_h))
        jitter_px = int(training_cfg.get("jitter_px", 0))
        x = avatar.shape[1] - new_w + self.jitter(jitter_px)
        y = avatar.shape[0] - new_h + self.jitter(jitter_px)
        return overlay_rgba(out, training_icon, x, y)

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
        if self.train and augment_cfg.get("background_drift"):
            background = random_color_drift(background, augment_cfg["background_drift"], self.rng)

        avatar = self.compose_overlays(read_rgba(avatar_path), row)
        compose_cfg = self.cfg.get("compose", {})
        avatar_base_scale = float(compose_cfg.get("avatar_scale", 1.0))
        avatar_offset = compose_cfg.get("avatar_offset", [0, 0])
        base_shift_x = int(avatar_offset[0]) if len(avatar_offset) > 0 else 0
        base_shift_y = int(avatar_offset[1]) if len(avatar_offset) > 1 else 0
        if self.train:
            scale = avatar_base_scale * float(self.rng.uniform(*augment_cfg.get("scale_range", [1.0, 1.0])))
            translate = int(augment_cfg.get("translate_px", 0))
            shift_x = base_shift_x + (int(self.rng.integers(-translate, translate + 1)) if translate > 0 else 0)
            shift_y = base_shift_y + (int(self.rng.integers(-translate, translate + 1)) if translate > 0 else 0)
        else:
            scale = avatar_base_scale
            shift_x = base_shift_x
            shift_y = base_shift_y

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
                corners = mask_cfg.get("train_corners", mask_cfg.get("corners", ["top_left", "top_right", "bottom_right"]))
            else:
                weight = float(mask_cfg.get("inference_weight", 0.2))
                corners = mask_cfg.get("inference_corners", mask_cfg.get("corners", ["top_left", "top_right", "bottom_right"]))
            image = apply_corner_soft_mask(
                image,
                grid=int(mask_cfg.get("grid", 4)),
                corners=corners,
                weight=weight,
            )
        return image
