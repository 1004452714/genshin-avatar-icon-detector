from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import AvatarNet


def load_model_from_checkpoint(path: str | Path, device: torch.device) -> tuple[AvatarNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["config"]
    mappings = checkpoint["mappings"]
    model_cfg = cfg.get("model", {})
    class_mapping = mappings.get("variant_to_idx", mappings.get("appearance_to_idx", {}))
    model = AvatarNet(
        num_appearances=len(class_mapping),
        num_rarities=len(mappings["rarity_to_idx"]) if model_cfg.get("rarity_head", True) else 0,
        embedding_dim=int(model_cfg.get("embedding_dim", 64)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        metric_head=model_cfg.get("metric_head"),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint
