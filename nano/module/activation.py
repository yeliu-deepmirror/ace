"""
https://github.com/RangiLyu/nanodet/blob/480ee5ae83341e09f3e3e7939dfd13d8e301a2cc/nanodet/model/module/activation.py
"""
import torch.nn as nn

activations = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ReLU6": nn.ReLU6,
    "SELU": nn.SELU,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "PReLU": nn.PReLU,
    None: nn.Identity,
}


def act_layers(name):
    assert name in activations.keys()
    if name == "LeakyReLU":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif name == "GELU":
        return nn.GELU()
    elif name == "PReLU":
        return nn.PReLU()
    else:
        return activations[name](inplace=True)
