import copy
import torch
from torch import nn, Tensor
from shufflenetv2 import build_shufflenetv2_backbone
from ghost_pan import build_ghostpan_fpn
from nanodet_plus_head import build_head


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
        self.head = build_head(nano_config["head_cfg"])


    def forward(self, x: torch.Tensor) -> Tensor:
        bb_feats = self.backbone(x)
        fpn_feats = self.fpn(bb_feats)
        # print(len(fpn_feats))
        head_out = self.head(feats=fpn_feats)
        return head_out



def save_model(model, path, epoch, iter, optimizer=None):
    print("save model to", path)
    model_state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )
    data = {"epoch": epoch, "state_dict": model_state_dict, "iter": iter}
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()
    torch.save(data, path)



def load_model_weight(model, model_path):
    print("load model from", model_path)
    checkpoint = torch.load(model_path)
    state_dict = checkpoint["state_dict"].copy()
    for k in checkpoint["state_dict"]:
        # convert average model weights
        if k.startswith("avg_model."):
            v = state_dict.pop(k)
            state_dict[k[4:]] = v
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if list(state_dict.keys())[0].startswith("model."):
        state_dict = {k[6:]: v for k, v in state_dict.items()}

    model_state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                state_dict[k] = model_state_dict[k]
        else:
            logger.log("Drop parameter {}.".format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print("No param", k)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)
