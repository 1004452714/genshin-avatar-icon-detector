from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from .config import load_config, project_path
from .dataset import load_labels, resolve_data_path


def validate_data(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    root = Path(cfg["_project_root"])
    labels_path = project_path(cfg, cfg["data"]["labels_csv"])
    df = load_labels(labels_path)
    errors = []
    warnings = []
    for i, row in df.iterrows():
        avatar_path = resolve_data_path(root, row["image_path"])
        bg_path = resolve_data_path(root, row["background_path"])
        if avatar_path is None or not avatar_path.exists():
            errors.append(f"第 {i} 行: 找不到角色图: {avatar_path}")
        else:
            try:
                with Image.open(avatar_path) as img:
                    if "A" not in img.getbands():
                        warnings.append(f"第 {i} 行: 角色图没有 alpha 通道: {avatar_path}")
            except Exception as exc:
                errors.append(f"第 {i} 行: 无法读取角色图 {avatar_path}: {exc}")
        if bg_path is not None and not bg_path.exists():
            errors.append(f"第 {i} 行: 找不到背景图: {bg_path}")
    print(f"行数={len(df)}")
    for warning in warnings:
        print(f"警告: {warning}")
    if errors:
        for error in errors:
            print(f"错误: {error}")
        raise SystemExit(1)
    print("数据校验通过")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_data(args.config)
