"""
from https://github.com/RangiLyu/nanodet/blob/a59db3c77b59ee0efb5c42aba05aed09fd9cefab/nanodet/model/head/nanodet_plus_head.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa N812
from typing import Tuple, Dict, Optional
from module.conv import ConvModule, Scale
from functools import partial


def build_head(aux_head_cfg):
    print("=> build aux head", aux_head_cfg)
    cfg_args = aux_head_cfg.copy()
    func_args = {}
    func_args.update(cfg_args)
    return SimpleConvHead(**func_args)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class SimpleConvHead(nn.Module):
    # GN snpe has problem
    def __init__(
            self,
            num_classes,
            input_channel,
            feat_channels=256,
            stacked_convs=4,
            strides=[8, 16, 32],  # noqa
            conv_cfg=None,
            norm_cfg=dict(type="BN"),  # noqa
            act_cfg="LeakyReLU",  # noqa
            reg_max=16,
            **kwargs):
        super(SimpleConvHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.reg_max = reg_max

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.init_layers()
        self.init_weights()

    def init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            stride = 1 if i < 3 else 2
            # stride = 1
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=stride,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ))
        self.gfl_reg = nn.Conv2d(self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
        # SNPE 1.6 has bug with Scale, so we remove this layer
        # self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

        # the output value ranges (0, 1), but Sigmoid might be too strong
        self.fc1 = nn.Linear(13632, 50)
        self.fc1_act = nn.Sigmoid()
        # self.normal_value = torch.tensor(1.2, dtype=torch.float)
        # self.offset_value = torch.tensor(-0.1, dtype=torch.float)


    def init_weights(self):
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        normal_init(self.gfl_reg, std=0.01)
        normal_init(self.fc1, std=0.01)

    def forward(self, feats):
        outputs = []
        for x in feats:
            reg_feat = x
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            bbox_pred = self.gfl_reg(reg_feat).float()
            output = bbox_pred.flatten(start_dim=2)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=2).flatten(start_dim=1)
        # outputs = self.normal_value * self.fc1_act(self.fc1(outputs)) - self.offset_value
        outputs = self.fc1_act(self.fc1(outputs))
        outputs = outputs.view(-1, 10, 5)
        return outputs
