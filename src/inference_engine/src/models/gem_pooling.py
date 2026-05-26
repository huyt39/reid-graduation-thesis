"""GeM (Generalized Mean) spatial pooling for ReID embeddings.

PDF Bước 4 specifies GeM pooling instead of plain global average pooling.
At inference time GeM(p=3) reduces a (B, C, H, W) feature map to a (B, C)
vector via ``adaptive_avg_pool2d(x.pow(p)).pow(1/p)``. With p=3 the
network emphasises the most discriminative spatial responses without
collapsing to a hard maximum the way max-pooling does, which improves
ReID rank-1 by 1–3% on standard benchmarks vs avgpool.

``p`` is stored as a non-trainable buffer by default so loading
checkpoints trained with ``nn.AdaptiveAvgPool2d`` still works — the
buffer is initialised from the constructor and doesn't appear under
the original pool's key in the state_dict, so the strict matcher in
``load_pretrained_weights`` skips it cleanly.
"""
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
        return pooled
