
import math
import torch
from ultralytics import YOLO
import cv2


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



def load_image(image_raw):
    image = cv2.resize(image_raw, (640, 640))
    image_tensor = torch.from_numpy(image)
    image_tensor = torch.permute(image_tensor, (2, 0, 1)).unsqueeze(0).float() / 255
    return image_tensor


def plot_image(image_show, preds):
    ratio_x = image_show.shape[1] / 640
    ratio_y = image_show.shape[0] / 640
    for i, pred in enumerate(preds):
        pred = pred.cpu().detach().numpy()
        # print(i, pred)
        x1 = int(pred[0] * ratio_x)
        y1 = int(pred[1] * ratio_y)
        x2 = int(pred[2] * ratio_x)
        y2 = int(pred[3] * ratio_y)

        color = (255, 0, 0)
        image_show = cv2.putText(image_show, str(int(pred[5])) + " - " + str(int(pred[4] * 100)),
                                (x1, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX ,
                                 0.7, (255, 0, 255), 2, cv2.LINE_AA)

        color = (0, 255, 0)
        image_show = cv2.line(image_show, (x1, y1), (x1, y2), color, 2)
        image_show = cv2.line(image_show, (x1, y1), (x2, y1), color, 2)
        image_show = cv2.line(image_show, (x1, y2), (x2, y2), color, 2)
        image_show = cv2.line(image_show, (x2, y1), (x2, y2), color, 2)
    return image_show


def modify_coord(x):
    if x < 0:
        return 0
    if x >= 640:
        return 640
    return x


def get_label_blocks(image_raw, preds, kept_labels={56, 67, 12}):
    ratio_x = image_raw.shape[1] / 640
    ratio_y = image_raw.shape[0] / 640
    sub_images = []
    for i, pred in enumerate(preds):
        pred = pred.cpu().detach().numpy()
        label = int(pred[5])
        if not label in kept_labels:
            continue
        # print(i, pred)
        x1 = int(modify_coord(pred[0]) * ratio_x)
        y1 = int(modify_coord(pred[1]) * ratio_y)
        x2 = int(modify_coord(pred[2]) * ratio_x)
        y2 = int(modify_coord(pred[3]) * ratio_y)

        sub_image = image_raw[y1:y2, x1:x2, :]
        sub_images.append(sub_image)
    return sub_images
