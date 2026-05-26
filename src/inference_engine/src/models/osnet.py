"""OSNet — Omni-Scale Network for Person Re-Identification.

Ported from reid-microservices/src/model_serving/src/models/osnet.py.
Changes from original: removed global-level device detection and FileNotFoundError;
weight paths are now passed explicitly to the factory function.
"""
from __future__ import absolute_import, division

from collections import OrderedDict
import sys

import numpy as np
import structlog
import torch
from torch import nn
from torch.nn import functional as F

from .gem_pooling import GeM

log = structlog.get_logger()

__all__ = ["osnet_x1_0"]


# ── Basic layers ─────────────────────────────────────────────────────

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if IN else nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LightConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


# ── Omni-scale building blocks ───────────────────────────────────────

class ChannelGate(nn.Module):
    def __init__(self, in_channels, num_gates=None, return_gates=False, gate_activation="sigmoid", reduction=16, layer_norm=False):
        super().__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1)) if layer_norm else None
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == "sigmoid":
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == "relu":
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == "linear":
            self.gate_activation = None
        else:
            raise RuntimeError(f"Unknown gate activation: {gate_activation}")

    def forward(self, x):
        inp = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        return x if self.return_gates else inp * x


class OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, IN=False, bottleneck_reduction=4, **kwargs):
        super().__init__()
        mid = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_channels)
        self.downsample = Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None
        self.IN = nn.InstanceNorm2d(out_channels, affine=True) if IN else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


# ── OSNet ─────────────────────────────────────────────────────────────

class OSNet(nn.Module):
    def __init__(self, num_classes, blocks, layers, channels, feature_dim=512, loss="softmax", IN=False, **kwargs):
        super().__init__()
        self.loss = loss
        self.feature_dim = feature_dim

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1], reduce_spatial_size=True, IN=IN)
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        # PDF Bước 4 — GeM(p=3) replaces plain avgpool for the
        # spatial→vector reduction. The buffer-based ``p`` is initialised
        # at construction; pretrained checkpoints (which contain no
        # state for nn.AdaptiveAvgPool2d) still load cleanly because the
        # loader at load_pretrained_weights() ignores keys absent from
        # the checkpoint. Name kept as ``global_avgpool`` so downstream
        # forward() and any saved-with-this-class checkpoints stay
        # backward-compatible.
        self.global_avgpool = GeM(p=3.0, learnable=False)
        self.fc = self._construct_fc_layer(self.feature_dim, channels[3], dropout_p=None)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(self, block, layer, in_ch, out_ch, reduce_spatial_size, IN=False):
        layers = [block(in_ch, out_ch, IN=IN)]
        for _ in range(1, layer):
            layers.append(block(out_ch, out_ch, IN=IN))
        if reduce_spatial_size:
            layers.append(nn.Sequential(Conv1x1(out_ch, out_ch), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]
        layers = []
        for dim in fc_dims:
            layers.extend([nn.Linear(input_dim, dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True)])
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == "softmax":
            return y
        elif self.loss == "triplet":
            return y, v
        raise KeyError(f"Unsupported loss: {self.loss}")


# ── Weight loading ────────────────────────────────────────────────────

def load_pretrained_weights(model: nn.Module, weight_path: str, device: torch.device) -> None:
    # Some legacy checkpoints were pickled against NumPy module paths that are
    # no longer importable under newer versions. Register a compatibility alias
    # before torch.load so those checkpoints can still be deserialized.
    sys.modules.setdefault("numpy._core", np.core)
    state_dict = torch.load(weight_path, map_location=device, weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched, discarded = [], []
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched.append(k)
        else:
            discarded.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    log.info("osnet.weights_loaded", matched=len(matched), discarded=len(discarded), path=weight_path)


# ── Factory ───────────────────────────────────────────────────────────

def osnet_x1_0(
    num_classes: int = 1000,
    loss: str = "softmax",
    weight_path: str | None = None,
    device: torch.device | None = None,
) -> OSNet:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
    ).to(device)
    if weight_path:
        load_pretrained_weights(model, weight_path, device)
    return model
