import copy
import torch
from torch import nn, Tensor
from shufflenetv2 import build_shufflenetv2_backbone
from ghost_pan import build_ghostpan_fpn


  # aux_head_cfg:
  #   type: SimpleConvHead
  #   num_classes: 1
  #   input_channel: 192
  #   feat_channels: 192
  #   stacked_convs: 8
  #   strides: [8, 16, 32, 64]
  #   act_cfg: LeakyReLU
  #   reg_max: 7
  #
class SimpleConvHead(nn.Module):

    def __init__(
            self,
            num_classes,
            input_channel,
            feat_channels=256,
            stacked_convs=4,
            strides=[8, 16, 32],  # noqa
            conv_cfg=None,
            norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),  # noqa
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
        self.cls_out_channels = num_classes

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ))
        self.gfl_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.gfl_reg = nn.Conv2d(self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = -4.595
        normal_init(self.gfl_cls, std=0.01, bias=bias_cls)
        normal_init(self.gfl_reg, std=0.01)

    def forward(self, feats):
        outputs = []
        for x, scale in zip(feats, self.scales):
            cls_feat = x
            reg_feat = x
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            cls_score = self.gfl_cls(cls_feat)
            bbox_pred = scale(self.gfl_reg(reg_feat)).float()
            output = torch.cat([cls_score, bbox_pred], dim=1)
            outputs.append(output.flatten(start_dim=2))
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs


class NanoLines(nn.Module):
    """Change nano det model to detect line features

        * backbone : using ShuffleNetV2
        * fpn : using GhostPAN
        * head : using NanoDetPlusHead
        * aux : using SimpleConvHead
        * post_process : since each line shall only have one output, we skip the NMS post process.

    """

    def __init__(self,
                 nano_config):
        super().__init__()

        self.backbone = build_shufflenetv2_backbone(nano_config["backbone_cfg"])
        self.fpn = build_ghostpan_fpn(nano_config["fpn_cfg"])
        self.head = build_nanodet_plus_head(nano_config["head_cfg"])
        self.detach_epoch = detach_epoch


    def forward(self, x: torch.Tensor) -> Tensor:
        im_shape = x.shape[2:]  # HxW
        bb_feats = self.backbone(x)
        fpn_feats = self.fpn(bb_feats)
        # head_out = self.head(feats=fpn_feats, im_shape=im_shape)

        return head_out
