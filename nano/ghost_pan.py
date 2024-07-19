"""
from https://github.com/RangiLyu/nanodet/blob/dcc7e798adbb9e64c406173a0792b934dbd76497/nanodet/model/fpn/ghost_pan.py
"""
import torch
import torch.nn as nn
from typing import List, Dict
from loguru import logger
from ghostnet import GhostBottleneck

from module.conv import ConvModule, DepthwiseConvModule, init_weights
from ghostnet import GhostBottleneck


def build_ghostpan_fpn(fpn_cfg):
    print("=> build ghostpan fpn", fpn_cfg)
    cfg_args = fpn_cfg.copy()
    func_args = {}
    func_args.update(cfg_args)
    return GhostPAN(**func_args)



class GhostBlocks(nn.Module):
    """Stack of GhostBottleneck used in GhostPAN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expand (int): Expand ratio of GhostBottleneck. Default: 1.
        kernel_size (int): Kernel size of depthwise convolution. Default: 5.
        num_blocks (int): Number of GhostBottlecneck blocks. Default: 1.
        use_res (bool): Whether to use residual connection. Default: False.
        activation (str): Name of activation function. Default: LeakyReLU.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expand=1,
                 kernel_size=5,
                 num_blocks=1,
                 use_res=False,
                 act_cfg=None):
        super().__init__()
        if act_cfg is None:
            act_cfg = "LeakyReLU"
        self.use_res = use_res
        if use_res:
            self.reduce_conv = ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                act_cfg=act_cfg,
            )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                GhostBottleneck(
                    in_channels,
                    int(out_channels * expand),
                    out_channels,
                    dw_kernel_size=kernel_size,
                    act_cfg=act_cfg,
                ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.blocks(x)
        if self.use_res:
            out = out + self.reduce_conv(x)
        return out


class GhostPAN(nn.Module):
    """Path Aggregation Network with Ghost block.

    Args:
        in_channels: Number of input channels per scale.
        out_channels: Number of output channels (used at each scale)
        use_depthwise: Whether to depthwise separable convolution in blocks. Default: False
        kernel_size: Kernel size of depthwise convolution. Default: 5.
        expand: Expand ratio of GhostBottleneck. Default: 1.
        num_blocks: Number of GhostBottlecneck blocks. Default: 1.
        use_res: Whether to use residual connection. Default: False.
        num_extra_level: Number of extra conv layers for more feature levels.Default: 0.
        upsample_cfg: Default: `dict(scale_factor=2, mode='nearest')`
        norm_cfg: Config for normalization layer.
        act_cfg: Activation layer config.
            Default: LeakyReLU.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        use_depthwise: bool = False,
        kernel_size: int = 5,
        expand: int = 1,
        num_blocks: int = 1,
        use_res: bool = False,
        num_extra_level: int = 0,
        upsample_cfg: Dict = None,
        norm_cfg: Dict = None,
        act_cfg: str = None,
    ):
        super(GhostPAN, self).__init__()
        assert num_extra_level >= 0
        assert num_blocks >= 1
        self.in_channels = in_channels
        self.out_channels = out_channels

        if act_cfg is None:
            act_cfg = "LeakyReLU"
        if norm_cfg is None:
            norm_cfg = {"type": "BN"}
        if upsample_cfg is None:
            upsample_cfg = {"scale_factor": 2, "mode": "bilinear"}

        conv = DepthwiseConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    out_channels,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ))
        self.top_down_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.top_down_blocks.append(
                GhostBlocks(
                    out_channels * 2,
                    out_channels,
                    expand,
                    kernel_size=kernel_size,
                    num_blocks=num_blocks,
                    use_res=use_res,
                    act_cfg=act_cfg,
                ))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ))
            self.bottom_up_blocks.append(
                GhostBlocks(
                    out_channels * 2,
                    out_channels,
                    expand,
                    kernel_size=kernel_size,
                    num_blocks=num_blocks,
                    use_res=use_res,
                    act_cfg=act_cfg,
                ))

        # extra layers
        self.extra_lvl_in_conv = nn.ModuleList()
        self.extra_lvl_out_conv = nn.ModuleList()
        for _ in range(num_extra_level):
            self.extra_lvl_in_conv.append(
                conv(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ))
            self.extra_lvl_out_conv.append(
                conv(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ))
        print("normal init ghost pan")
        self.apply(init_weights)

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.
        Returns:
            tuple[Tensor]: multi level features.
        """
        assert len(inputs) == len(self.in_channels)
        inputs = [reduce(input_x) for input_x, reduce in zip(inputs, self.reduce_layers)]
        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]

            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](torch.cat(
                [upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # extra layers
        for extra_in_layer, extra_out_layer in zip(self.extra_lvl_in_conv, self.extra_lvl_out_conv):
            outs.append(extra_in_layer(inputs[-1]) + extra_out_layer(outs[-1]))

        return tuple(outs)
