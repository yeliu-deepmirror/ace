"""
from https://github.com/RangiLyu/nanodet/blob/a59db3c77b59ee0efb5c42aba05aed09fd9cefab/nanodet/model/head/nanodet_plus_head.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa N812
from typing import Tuple, Dict, Optional
from module.conv import ConvModule, DepthwiseConvModule, Scale
from module.dsl_assigner import DynamicSoftLabelAssigner
from functools import partial


def build_nanodet_plus_head(head_cfg):
    print("=> build nanodet_plus head", head_cfg)
    cfg_args = head_cfg.copy()
    func_args = {}
    func_args.update(cfg_args)
    return NanoDetPlusHead(**func_args)


def build_aux_head(aux_head_cfg):
    print("=> build aux head", aux_head_cfg)
    cfg_args = aux_head_cfg.copy()
    func_args = {}
    func_args.update(cfg_args)
    return SimpleConvHead(**func_args)


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))



def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def distance2bbox(points: torch.Tensor, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points: Shape (n, 2), [x, y].
        distance: Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape: Shape of the image, [H, w].

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer("project", torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        shape = x.size()
        x = F.softmax(x.reshape(*shape[:-1], 4, self.reg_max + 1), dim=-1)
        x = F.linear(x, self.project.type_as(x)).reshape(*shape[:-1], 4)
        return x


class NanoDetPlusHead(nn.Module):

    def __init__(
            self,
            num_classes: int,
            input_channel,
            feat_channels=96,
            stacked_convs=2,
            kernel_size=5,
            strides=None,
            conv_type="DWConv",
            norm_cfg=dict(type="BN"),  # noqa
            reg_max=7,
            act_cfg="LeakyReLU",  # noqa
            assigner_cfg=None,  # noqa
            **kwargs):
        super().__init__()
        if strides is None:
            strides = [8, 16, 32, 64]
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_size = kernel_size
        self.reg_max = reg_max
        self.act_cfg = act_cfg
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule
        self.norm_cfg = dict(norm_cfg)
        self.strides = strides

        # init module
        self.cls_convs = nn.ModuleList()
        for _ in self.strides:
            cls_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)

        self.gfl_cls = nn.ModuleList([
            nn.Conv2d(
                self.feat_channels,
                self.num_classes + 4 * (self.reg_max + 1),
                1,
                padding=0,
            ) for _ in self.strides
        ])
        if assigner_cfg is not None:
            self.assigner = DynamicSoftLabelAssigner(**assigner_cfg)
        self.distribution_project = Integral(self.reg_max)
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
                ))
        return cls_convs

    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
        print("Finish initialize NanoDet-Plus Head.")

    def get_single_level_center_priors(self, batch_size: int, featmap_size: Tuple[int], stride: int,
                                       device: torch.device):
        """Generate centers of a single stage feature map.
        Args:
            batch_size: Number of images in one batch.
            featmap_size: height and width of the feature map
            stride (int): down sample stride of the feature map
            dtype: data type of the tensors
            device: device of the tensors
        Return:
            priors (Tensor): center priors of a single level feature map.
        """
        dtype = torch.float32
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
        y, x = torch.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0], ), stride)
        proiors = torch.stack([x, y, strides, strides], dim=-1)
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1)

    def forward_onnx(self, feats):
        """only used for onnx export"""
        outputs = []
        for feat, cls_convs, gfl_cls in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
        ):
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            cls_pred, reg_pred = output.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=1
            )
            cls_pred = cls_pred.sigmoid()
            out = torch.cat([cls_pred, reg_pred], dim=1)
            outputs.append(out.flatten(start_dim=2))
        return torch.cat(outputs, dim=2).permute(0, 2, 1)

    def forward(self, feats, im_shape):
        if torch.onnx.is_in_onnx_export():
            return self.forward_onnx(feats)
        outputs = []
        featmap_sizes = []
        for feat, cls_convs, gfl_cls in zip(feats, self.cls_convs, self.gfl_cls):
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            featmap_sizes.append(output.shape[2:])
            outputs.append(output.flatten(start_dim=2))
        preds = torch.cat(outputs, dim=2).permute(0, 2, 1)
        cls_preds, reg_preds = preds.split([self.num_classes, 4 * (self.reg_max + 1)], dim=-1)

        batch_size, _, _ = preds.shape
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                batch_size,
                featmap_sizes[i],
                stride,
                device=preds.device,
            ) for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]

        if not self.training:
            decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=im_shape)
            scores = cls_preds.sigmoid()
            return scores, decoded_bboxes
        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)
        return cls_preds, reg_preds, decoded_bboxes, center_priors

    def train_post_process(self, center_priors, aux_preds, gt_meta: Dict):
        aux_cls_preds, aux_reg_preds = aux_preds.split([self.num_classes, 4 * (self.reg_max + 1)],
                                                       dim=-1)
        aux_dis_preds = (self.distribution_project(aux_reg_preds) * center_priors[..., 2, None])
        aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds)
        batchsize, _, _ = aux_preds.shape
        gt_bboxes_ignore = gt_meta["gt_bboxes_ignore"]
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(batchsize)]

        batch_assign_res = multi_apply(self.target_assign_single_img,
                                       aux_cls_preds.detach(), center_priors,
                                       aux_decoded_bboxes.detach(), gt_meta['gt_bboxes'],
                                       gt_meta['gt_labels'], gt_bboxes_ignore)
        return aux_cls_preds, aux_reg_preds, aux_decoded_bboxes, batch_assign_res

    @torch.no_grad()
    def target_assign_single_img(
        self,
        cls_preds: torch.Tensor,
        center_priors: torch.Tensor,
        decoded_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes_ignore: Optional[torch.Tensor] = None,
    ):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds: Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            center_priors: All priors of one image, a 2D-Tensor with
                shape [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes: Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes: Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels: Ground truth labels of one image, a Tensor
                with shape [num_gts].
            gt_bboxes_ignore: Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
        """

        device = center_priors.device
        gt_bboxes = gt_bboxes.to(device)
        gt_labels = gt_labels.to(device)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)

        if gt_bboxes_ignore is not None:
            gt_bboxes_ignore = gt_bboxes_ignore.to(device)
            gt_bboxes_ignore = gt_bboxes_ignore.to(decoded_bboxes.dtype)

        assign_result = self.assigner.assign(
            cls_preds.sigmoid(),
            center_priors,
            decoded_bboxes,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
        )
        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes)

        num_priors = center_priors.size(0)
        bbox_targets = torch.zeros_like(center_priors)
        dist_targets = torch.zeros_like(center_priors)
        labels = center_priors.new_full((num_priors, ), self.num_classes, dtype=torch.long)
        label_weights = center_priors.new_zeros(num_priors, dtype=torch.float)
        label_scores = center_priors.new_zeros(labels.shape, dtype=torch.float)

        num_pos_per_img = pos_inds.size(0)
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            bbox_targets[pos_inds, :] = pos_gt_bboxes
            dist_targets[pos_inds, :] = (bbox2distance(center_priors[pos_inds, :2], pos_gt_bboxes) /
                                         center_priors[pos_inds, None, 2])
            dist_targets = dist_targets.clamp(min=0, max=self.reg_max - 0.1)
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        return (labels, label_scores, label_weights, bbox_targets, dist_targets, num_pos_per_img)

    def sample(self, assign_result, gt_bboxes):
        """Sample positive and negative bboxes."""
        pos_inds = (torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique())
        neg_inds = (torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique())
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds


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
