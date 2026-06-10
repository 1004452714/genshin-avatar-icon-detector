from __future__ import annotations

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
    ) -> None:
        super().__init__()
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
        self.classifier = nn.Linear(embedding_dim, num_appearances)
        self.rarity_head = nn.Linear(embedding_dim, num_rarities) if num_rarities > 0 else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features(x)
        embedding = F.normalize(self.embedding(features), dim=1)
        class_logits = self.classifier(embedding)
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

