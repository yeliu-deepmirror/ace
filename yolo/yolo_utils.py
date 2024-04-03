import math
import torch
from ultralytics import YOLO



class SiluT(torch.nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(input)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def update_conv(conv):
    # print(conv.act)
    conv.act = SiluT(inplace=True)

def update_silu_in_model(network, prefix=""):
    if network.__class__.__name__ == 'Conv':
        update_conv(network)
        return

    if network.__class__.__name__ == 'C2f':
        update_conv(network.cv1)
        update_conv(network.cv2)
        for i in range(len(network.m)):
            update_conv(network.m[i].cv1)
            update_conv(network.m[i].cv2)
        return

    if network.__class__.__name__ == 'SPPF':
        update_conv(network.cv1)
        update_conv(network.cv2)
        return

    if network.__class__.__name__ == 'Detect':
        update_silu_in_model(network.cv2)
        update_silu_in_model(network.cv3)
        return

    if network.__class__.__name__ == 'ModuleList':
        for i in range(len(network)):
            update_silu_in_model(network[i])

    if network.__class__.__name__ == 'Sequential':
        for i in range(len(network)):
            update_silu_in_model(network[i])
