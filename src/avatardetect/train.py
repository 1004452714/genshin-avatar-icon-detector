from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config, project_path
from .dataset import AvatarDataset, load_labels, make_mappings, split_labels
from .model import AvatarNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[str, dict[str, int]]]:
    root = Path(cfg["_project_root"])
    df = load_labels(project_path(cfg, cfg["data"]["labels_csv"]))
    train_df, val_df = split_labels(df)
    mappings = make_mappings(df)
    seed = int(cfg["train"].get("seed", 42))
    train_set = AvatarDataset(train_df, root, cfg, mappings, train=True, seed=seed)
    val_set = AvatarDataset(val_df, root, cfg, mappings, train=False, seed=seed + 1)
    batch_size = int(cfg["train"].get("batch_size", 64))
    num_workers = int(cfg["data"].get("num_workers", 0))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, mappings


def run_epoch(
    model: AvatarNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    cfg: dict[str, Any],
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    rarity_weight = float(cfg["train"].get("rarity_loss_weight", 0.0))
    use_amp = bool(cfg["train"].get("mixed_precision", True)) and device.type == "cuda"

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        appearance_idx = batch["appearance_idx"].to(device, non_blocking=True)
        rarity_idx = batch["rarity_idx"].to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast(enabled=use_amp):
                _, class_logits, rarity_logits = model(images)
                loss = ce(class_logits, appearance_idx)
                if rarity_weight > 0 and rarity_logits.shape[1] > 0:
                    valid = rarity_idx.ge(0)
                    if valid.any():
                        loss = loss + rarity_weight * ce(rarity_logits[valid], rarity_idx[valid])

            if is_train:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.detach().cpu()) * images.shape[0]
        total_correct += int(class_logits.argmax(dim=1).eq(appearance_idx).sum().detach().cpu())
        total_count += int(images.shape[0])

    return {
        "loss": total_loss / max(1, total_count),
        "top1": total_correct / max(1, total_count),
    }


def save_checkpoint(
    path: Path,
    model: AvatarNet,
    mappings: dict[str, dict[str, int]],
    cfg: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "mappings": mappings,
        "config": cfg,
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(payload, path)


def train_main(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    seed_everything(int(cfg["train"].get("seed", 42)))
    train_loader, val_loader, mappings = build_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg.get("model", {})
    model = AvatarNet(
        num_appearances=len(mappings["appearance_to_idx"]),
        num_rarities=len(mappings["rarity_to_idx"]) if model_cfg.get("rarity_head", True) else 0,
        embedding_dim=int(model_cfg.get("embedding_dim", 64)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("learning_rate", 1e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    scaler = GradScaler(enabled=device.type == "cuda" and bool(cfg["train"].get("mixed_precision", True)))
    save_dir = project_path(cfg, cfg["train"].get("save_dir", "outputs/checkpoints"))
    best_top1 = -1.0

    print(f"设备={device}")
    print(f"外观数量={len(mappings['appearance_to_idx'])} 稀有度数量={len(mappings['rarity_to_idx'])}")
    for epoch in range(1, int(cfg["train"].get("epochs", 80)) + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, scaler, cfg)
        val_metrics = run_epoch(model, val_loader, device, None, None, cfg)
        print(
            f"轮次={epoch} "
            f"训练损失={train_metrics['loss']:.4f} 训练top1={train_metrics['top1']:.4f} "
            f"验证损失={val_metrics['loss']:.4f} 验证top1={val_metrics['top1']:.4f}"
        )
        save_checkpoint(save_dir / "last.pt", model, mappings, cfg, epoch, val_metrics)
        if val_metrics["top1"] >= best_top1:
            best_top1 = val_metrics["top1"]
            save_checkpoint(save_dir / "best.pt", model, mappings, cfg, epoch, val_metrics)
            (save_dir / "mappings.json").write_text(json.dumps(mappings, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_main(args.config)
