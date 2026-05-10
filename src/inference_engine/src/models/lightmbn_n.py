"""LMBN_n — Light Multi-Branch Network for Person Re-Identification.

Ported from reid-microservices/src/model_serving/src/models/lightmbn_n.py.
"""
import copy

import torch
from torch import nn

from .attention import BatchFeatureErase_Top
from .bnneck import BNNeck, BNNeck3
from .osnet import OSBlock, osnet_x1_0


class LMBN_n(nn.Module):
    def __init__(self, num_classes: int, feats: int, activation_map: bool,
                 osnet_weight_path: str | None = None, device: torch.device | None = None):
        super().__init__()

        self.n_ch = 2
        self.chs = 512 // self.n_ch

        osnet = osnet_x1_0(pretrained=False, weight_path=osnet_weight_path, device=device)

        self.backone = nn.Sequential(osnet.conv1, osnet.maxpool, osnet.conv2, osnet.conv3[0])
        conv3 = osnet.conv3[1:]

        self.global_branch = nn.Sequential(copy.deepcopy(conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))
        self.partial_branch = nn.Sequential(copy.deepcopy(conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))
        self.channel_branch = nn.Sequential(copy.deepcopy(conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        self.global_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.partial_pooling = nn.AdaptiveAvgPool2d((2, 1))
        self.channel_pooling = nn.AdaptiveAvgPool2d((1, 1))

        reduction = BNNeck3(512, num_classes, feats, return_f=True)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)

        self.shared = nn.Sequential(
            nn.Conv2d(self.chs, feats, 1, bias=False),
            nn.BatchNorm2d(feats),
            nn.ReLU(True),
        )
        self._weights_init_kaiming(self.shared)

        self.reduction_ch_0 = BNNeck(feats, num_classes, return_f=True)
        self.reduction_ch_1 = BNNeck(feats, num_classes, return_f=True)

        self.batch_drop_block = BatchFeatureErase_Top(512, OSBlock)
        self.activation_map = activation_map

        _dev = device or next(self.backone.parameters()).device
        self.to(_dev)

    def forward(self, x):
        x = self.backone(x)

        glo = self.global_branch(x)
        par = self.partial_branch(x)
        cha = self.channel_branch(x)

        if self.activation_map:
            glo_ = glo
            _, _, h_par, _ = par.size()
            return glo, glo_, par[:, :, :h_par // 2, :], par[:, :, h_par // 2:, :], cha[:, :self.chs, :, :], cha[:, self.chs:, :, :]

        if self.batch_drop_block is not None:
            glo_drop, glo = self.batch_drop_block(glo)

        glo_drop = self.global_pooling(glo_drop)
        glo = self.channel_pooling(glo)
        g_par = self.global_pooling(par)
        p_par = self.partial_pooling(par)
        cha = self.channel_pooling(cha)

        p0, p1 = p_par[:, :, 0:1, :], p_par[:, :, 1:2, :]

        f_glo = self.reduction_0(glo)
        f_p0 = self.reduction_1(g_par)
        f_p1 = self.reduction_2(p0)
        f_p2 = self.reduction_3(p1)
        f_glo_drop = self.reduction_4(glo_drop)

        c0, c1 = cha[:, :self.chs, :, :], cha[:, self.chs:, :, :]
        c0, c1 = self.shared(c0), self.shared(c1)
        f_c0 = self.reduction_ch_0(c0)
        f_c1 = self.reduction_ch_1(c1)

        if not self.training:
            return torch.stack([f_glo[0], f_glo_drop[0], f_p0[0], f_p1[0], f_p2[0], f_c0[0], f_c1[0]], dim=2)

        return [f_glo[1], f_glo_drop[1], f_p0[1], f_p1[1], f_p2[1], f_c0[1], f_c1[1]], [f_glo[-1], f_glo_drop[-1], f_p0[-1]]

    @staticmethod
    def _weights_init_kaiming(m):
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
            nn.init.constant_(m.bias, 0.0)
        elif classname.find("Conv") != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find("BatchNorm") != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
