import torch
import glob
import yaml
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler
import torchvision.transforms as transforms

from nano_plus import NanoLines, save_model, load_model_weight
from dataset.dataset import CarLinemarksDataset, LABEL_TO_ID, ID_TO_LABEL, transfrom_image_np

config_file = "nano/config/gray_config.yaml"
images_folder = "data/car_line_yvr"
model_path = "models/model_nano_lines.ckpt"
lines_foler = "data"

# class LineProjector:
#     def __init__(self, lines_foler):
#         self.line_points = np.zeros((len(LABEL_TO_ID), 6))
#         # read 3d lines from dataset
#         lines_file = lines_foler + "/lines.txt"
#         with open(lines_file, 'r') as file_in:
#             for line in file_in:
#                 message = line.split(',')
#                 idx = LABEL_TO_ID[message[6][:-1]]
#                 for i in range(6):
#                     self.line_points[idx, i] = float(message[i])
#         print(self.line_points)
#

with open(config_file) as stream:
    try:
        nano_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# run the test images
det_model = NanoLines(nano_config).cuda()
# load_model_weight(det_model, model_path)
det_model.load_state_dict(torch.load(model_path))

def draw_labels_to_image(image, label):
    image = torch.squeeze(image).cpu().detach().numpy()
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (480, 640))

    for i in range(label.shape[0]):
        if label[i, 0] < 0.5:
            continue
        start = (int(label[i, 1]), int(label[i, 2]))
        end = (int(label[i, 3]), int(label[i, 4]))
        color = (0, 1, 0)
        cv2.line(image, start, end, color, 3)

        center = (int(0.5 * (start[0] + end[0])), int(0.5 * (start[1] + end[1])))
        image = cv2.putText(image, str(i), center, cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 1, cv2.LINE_AA)


    plt.imshow(image)
    plt.show()

with torch.no_grad():
    images = glob.glob(images_folder + "/*.jpg")
    dataset = CarLinemarksDataset(images_folder)

    for idx in range(len(images)):
        image_cv = cv2.imread(images[idx])
        image = transfrom_image_np(image_cv)

        # image_pil = Image.open(images[idx])
        # image = dataset.transform(image_pil)

        outputs = det_model(image.unsqueeze(0).cuda())

        outputs = torch.squeeze(outputs).cpu().detach().numpy()
        outputs[:, 1:] = 640.0 * outputs[:, 1:]

        draw_labels_to_image(image, outputs)
