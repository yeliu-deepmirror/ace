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
            num_lines,
            input_channel,
            feat_channels=96,
            stacked_convs=2,
            kernel_size=5,
            strides=[8, 16, 32],  # noqa
            conv_cfg=None,
            norm_cfg=dict(type="BN"),  # noqa
            act_cfg="LeakyReLU",  # noqa
            reg_max=16,
            **kwargs):
        super(SimpleConvHead, self).__init__()
        self.num_lines = num_lines
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.kernel_size = kernel_size
        self.reg_max = reg_max
        self.ConvModule = ConvModule

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.init_layers()
        self.init_weights()

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    act_cfg=self.act_cfg,
                )
            )
        return cls_convs

    def init_layers(self):
        self.cls_convs = nn.ModuleList()
        for _ in self.strides:
            cls_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)

        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels,
                    96,
                    1,
                    padding=0,
                )
                for _ in self.strides
            ]
        )

        # the output value ranges (0, 1), but Sigmoid might be too strong
        self.fc1 = nn.Linear(163200, 50)
        self.fc1_act = nn.Sigmoid()
        # self.fc1_act = nn.LeakyReLU()
        # self.normal_value = torch.tensor(1.2, dtype=torch.float)
        # self.offset_value = torch.tensor(-0.1, dtype=torch.float)


    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # normal_init(self.fc1, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)


    def forward(self, feats):
        outputs = []
        for feat, cls_convs, gfl_cls in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
        ):
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            outputs.append(output.flatten(start_dim=2))
        # outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)

        # print(outputs.shape)
        # return outputs

        # outputs = self.final_pool(torch.cat(outputs, dim=2))
        outputs = torch.cat(outputs, dim=2).flatten(start_dim=1)

        print(outputs.shape)
        # outputs = self.normal_value * self.fc1_act(self.fc1(outputs)) - self.offset_value
        outputs = self.fc1_act(self.fc1(outputs))

        outputs = outputs.view(-1, 10, 5)
        return outputs
