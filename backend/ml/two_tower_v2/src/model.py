from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.backbone(x)
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding


class ItemTower(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.backbone(x)
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding


class TwoTowerModel(nn.Module):
    def __init__(self, user_input_dim: int, item_input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_tower = UserTower(user_input_dim, embedding_dim)
        self.item_tower = ItemTower(item_input_dim, embedding_dim)

    def forward(self, user_x: torch.Tensor, item_x: torch.Tensor) -> torch.Tensor:
        user_embedding = self.user_tower(user_x)
        item_embedding = self.item_tower(item_x)
        similarity = (user_embedding * item_embedding).sum(dim=-1, keepdim=True)
        return similarity

    def predict_score(self, user_x: torch.Tensor, item_x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(user_x, item_x)
        return torch.sigmoid(logits)

    def encode_user(self, user_x: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_x)

    def encode_item(self, item_x: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_x)
