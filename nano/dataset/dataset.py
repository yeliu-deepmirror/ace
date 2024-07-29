import glob
import random
import math
from PIL import Image
import numpy as np

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler
import torchvision.transforms as transforms

LABEL_TO_ID = {
    "right": 0,
    "left": 1,
    "bot": 2,
    "chair_bot": 3,
    "front": 4,
    "center": 5,
    "right_chair": 6,
    "left_chair": 7,
    "front_down": 8,
    "chair_top": 9,
}
ID_TO_LABEL = [
    "right",
    "left",
    "bot",
    "chair_bot",
    "front",
    "center",
    "right_chair",
    "left_chair",
    "front_down",
    "chair_top",
]


def transfrom_image_np(image_np):
    import cv2
    image_np = cv2.resize(image_np[:, :, 0], (256, 320))
    image_np = image_np.astype(np.float32) / 255.0
    image_np = np.expand_dims(image_np, axis=0)
    return torch.from_numpy(image_np)


class CarLinemarksDataset(Dataset):

    def __init__(self, data_set_folder):
        self.augmentation = False
        self.buffer = False
        self.images = glob.glob(data_set_folder + "/*/dataset/*.jpg")
        # TODO : rescale the image
        self.transform = transforms.Compose(
            [transforms.Grayscale(),
             transforms.ToTensor(),
             transforms.Resize((320, 256), antialias=True)])
        print("loaded", len(self.images), "data")
        self.num_labels = len(LABEL_TO_ID)
        self.data_buffer = []
        if self.buffer:
            print("=> buffering the whole dataset")
            for idx in range(len(self.images)):
                image = self.get_image(idx).cuda()
                label = torch.from_numpy(self.get_label(idx)).cuda()
                self.data_buffer.append([image, label])


    def __len__(self):
        return len(self.images)

    def get_image(self, idx):
        image = Image.open(self.images[idx])
        image_tensor = self.transform(image)
        return image_tensor

    def get_label(self, idx):
        labels = np.zeros((self.num_labels, 5))
        label_file = self.images[idx][:-4] + ".txt"
        with open(label_file, 'r') as file_in:
            for line in file_in:
                message = line.split(',')
                idx = LABEL_TO_ID[message[4][:-1]]
                labels[idx, 0] = 1.0
                for i in range(4):
                    labels[idx, i + 1] = float(message[i]) / 640
        return labels

    def rotate_labels(self, label, angle):
        theta = -angle * np.pi / 180.0
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta),  np.cos(theta)]])
        offset = np.array([240 / 640, 320 / 640])
        for idx in range(label.shape[0]):
            label[idx, 1:3] = np.dot(rot_matrix, label[idx, 1:3] - offset) + offset
            label[idx, 3:5] = np.dot(rot_matrix, label[idx, 3:5] - offset) + offset
        return label


    def __getitem__(self, idx):
        if self.buffer:
            tmp = self.data_buffer[idx]
            return (tmp[0], tmp[1])

        image = self.get_image(idx)
        label = self.get_label(idx)

        if self.augmentation:
            angle = 40.0 * (random.random() - 0.5)
            # rotate the image
            image = transforms.functional.rotate(image, angle, transforms.InterpolationMode.BILINEAR)
            # rotate the label
            label = self.rotate_labels(label, angle)

        return (image, label)
