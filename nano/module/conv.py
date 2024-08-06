"""
from https://github.com/RangiLyu/nanodet/blob/480ee5ae83341e09f3e3e7939dfd13d8e301a2cc/nanodet/model/module/conv.py
"""
import torch
import torch.nn as nn

from .norm import build_norm_layer
from .activation import act_layers


def init_weights(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d) and classname != "SplAtConv2d":
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init_range = 1.0 / math.sqrt(m.out_features)
        nn.init.uniform_(m.weight, -init_range, init_range)
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


class ConvModule(nn.Module):
    """A conv block that contains conv/norm/activation layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        activation (str): activation layer, "ReLU" by default.
        inplace (bool): Whether to use inplace mode for activation.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias="auto",
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None,
            inplace=True,
            order=("conv", "norm", "act"),
    ):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        if act_cfg is None:
            act_cfg = "ReLU"

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple)
        assert len(self.order) == 3
        assert set(order) == {"conv", "norm", "act"}

        self.with_norm = norm_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            logger.warning("ConvModule has norm and bias at the same time")

        # build convolution layer
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None

        # build activation layer
        if self.act_cfg:
            self.act = act_layers(self.act_cfg)

        # Use msra init by default
        self.apply(init_weights)

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def forward(self, x, norm=True):
        for layer in self.order:
            if layer == "conv":
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and self.act_cfg:
                x = self.act(x)
        return x


class DepthwiseConvModule(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias="auto",
            norm_cfg=None,
            act_cfg=None,
            inplace=True,
            order=("depthwise", "dwnorm", "act", "pointwise", "pwnorm", "act"),
    ):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = {"type": "BN"}

        if act_cfg is None:
            act_cfg = "ReLU"

        self.act_cfg = act_cfg
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple)
        assert len(self.order) == 6
        assert set(order) == {
            "depthwise",
            "dwnorm",
            "act",
            "pointwise",
            "pwnorm",
            "act",
        }

        self.with_norm = norm_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            logger.warning("ConvModule has norm and bias at the same time")

        # build convolution layer
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=bias)

        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.depthwise.in_channels
        self.out_channels = self.pointwise.out_channels
        self.kernel_size = self.depthwise.kernel_size
        self.stride = self.depthwise.stride
        self.padding = self.depthwise.padding
        self.dilation = self.depthwise.dilation
        self.transposed = self.depthwise.transposed
        self.output_padding = self.depthwise.output_padding

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            _, self.dwnorm = build_norm_layer(norm_cfg, in_channels)
            _, self.pwnorm = build_norm_layer(norm_cfg, out_channels)

        # build activation layer
        if self.act_cfg:
            self.act = act_layers(self.act_cfg)

        # Use msra init by default
        self.apply(init_weights)

    def forward(self, x, norm=True):
        for layer_name in self.order:
            if layer_name != "act":
                layer = self.__getattr__(layer_name)
                x = layer(x)
            elif layer_name == "act" and self.act_cfg:
                x = self.act(x)
        return x


class Scale(nn.Module):
    """
    A learnable scale parameter
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale
