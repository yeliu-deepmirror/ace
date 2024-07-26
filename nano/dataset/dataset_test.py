import torch
import glob
import yaml
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from dataset import CarLinemarksDataset, LABEL_TO_ID, ID_TO_LABEL, transfrom_image_np

data_set_folder = "/home/yeliu/Development/LidarMapping/data/map/"

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


dataset = CarLinemarksDataset(data_set_folder)

for idx in range(len(dataset)):
    image, label = dataset[idx]


    label[:, 1:] = 640.0 * label[:, 1:]
    draw_labels_to_image(image, label)
