from __future__ import annotations

import torch
from torch import nn


class DQN(nn.Module):
    """Manuscript-aligned vision-kinematic DQN.

    Input:
      semantic_stack: [B, 6, 300, 400] uint8 or float class-color frames
      speed: [B] or [B, 1] normalized scalar vehicle speed
    Output:
      45 Q-values
    """

    def __init__(self, num_actions: int = 45) -> None:
        super().__init__()
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(6, 96, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 192, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.speed_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.LayerNorm(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.LayerNorm(16),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(8192 + 16, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(512, num_actions),
        )

    def forward(self, semantic_stack: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        # Replay and environment supply uint8 semantic palette frames.
        # Avoid a GPU synchronization caused by inspecting x.max().item().
        x = semantic_stack.float() / 255.0 if semantic_stack.dtype == torch.uint8 else semantic_stack.float()
        if speed.ndim == 1:
            speed = speed.unsqueeze(1)
        visual = self.visual_encoder(x)
        speed_features = self.speed_encoder(speed.float())
        return self.head(torch.cat((visual, speed_features), dim=1))
