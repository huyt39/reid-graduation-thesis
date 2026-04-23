import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = False):
        super().__init__()
        if learnable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.tensor([p]))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_clamped = x.clamp(min=self.eps)
        pooled = F.adaptive_avg_pool2d(x_clamped.pow(self.p), 1).pow(1.0 / self.p)
        return pooled.view(x.size(0), -1)
