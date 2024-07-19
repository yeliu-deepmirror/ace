import torch
import glob
import yaml
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import random
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler
import torchvision.transforms as transforms

from nano_plus import NanoLines, save_model, load_model_weight
from dataset.dataset import CarLinemarksDataset

config_file = "nano/config/gray_config.yaml"
data_set_folder = "data/20240222T101812+0800_oppoma_/dataset"
model_path = "models/model_nano_lines_250.ckpt"


with open(config_file) as stream:
    try:
        nano_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
dataset = CarLinemarksDataset(data_set_folder)
test_loader = DataLoader(dataset, batch_size=1, sampler=SubsetRandomSampler(list(range(len(dataset)))))


# run the test images
det_model = NanoLines(nano_config)
load_model_weight(det_model, model_path)


def draw_labels_to_image(image, label_ref, label):
    image = torch.squeeze(image).numpy()
    label = torch.squeeze(label).detach().numpy()
    label_ref = torch.squeeze(label_ref).detach().numpy()
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (480, 640))

    for i in range(label_ref.shape[0]):
        if label_ref[i, 0] < 0.5:
            continue
        start = (int(label_ref[i, 1]), int(label_ref[i, 2]))
        end = (int(label_ref[i, 3]), int(label_ref[i, 4]))
        color = (1, 0, 0)
        cv2.line(image, start, end, color, 2)

        center = (int(0.5 * (start[0] + end[0])), int(0.5 * (start[1] + end[1])))
        image = cv2.putText(image, str(i), center, cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 1, cv2.LINE_AA)

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
    for i, data in enumerate(test_loader, 0):
        outputs = det_model(data[0])
        draw_labels_to_image(data[0], data[1], outputs)
        # break
