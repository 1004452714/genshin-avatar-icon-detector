from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class AvatarNet(nn.Module):
    def __init__(
        self,
        num_appearances: int,
        num_rarities: int,
        embedding_dim: int = 64,
        base_channels: int = 32,
        dropout: float = 0.1,
        metric_head: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        metric_cfg = metric_head or {}
        self.metric_head_enabled = bool(metric_cfg.get("enabled", False))
        self.metric_head_type = str(metric_cfg.get("type", "cosface")).lower()
        self.metric_scale = float(metric_cfg.get("scale", 30.0))
        self.metric_margin = float(metric_cfg.get("margin", 0.25))
        if self.metric_head_enabled and self.metric_head_type != "cosface":
            raise ValueError("metric_head.type 目前只支持 cosface")

        c = base_channels
        self.features = nn.Sequential(
            ConvBlock(3, c, stride=2),
            ConvBlock(c, c * 2, stride=2),
            ConvBlock(c * 2, c * 4, stride=2),
            ConvBlock(c * 4, c * 4, stride=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.embedding = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c * 4, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_appearances, bias=not self.metric_head_enabled)
        self.rarity_head = nn.Linear(embedding_dim, num_rarities) if num_rarities > 0 else None

    def classify(self, embedding: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        if not self.metric_head_enabled:
            return self.classifier(embedding)

        logits = F.linear(embedding, F.normalize(self.classifier.weight, dim=1))
        if targets is not None:
            one_hot = F.one_hot(targets, num_classes=logits.shape[1]).to(dtype=logits.dtype, device=logits.device)
            logits = logits - one_hot * self.metric_margin
        return logits * self.metric_scale

    @property
    def metric_output_scale(self) -> float:
        return self.metric_scale if self.metric_head_enabled else 1.0

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features(x)
        embedding = F.normalize(self.embedding(features), dim=1)
        class_logits = self.classify(embedding, targets)
        if self.rarity_head is None:
            rarity_logits = embedding.new_zeros((embedding.shape[0], 0))
        else:
            rarity_logits = self.rarity_head(embedding)
        return embedding, class_logits, rarity_logits


class OnnxAvatarWrapper(nn.Module):
    def __init__(self, model: AvatarNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding, _, rarity_logits = self.model(x)
        return embedding, rarity_logits
