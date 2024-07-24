import glob
import random
import cv2
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
    "right_b": 8,
    "left_b": 9,
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
    "right_b",
    "left_b",
]


def transfrom_image_np(image_np):
    image_np = cv2.resize(image_np[:, :, 0], (256, 320))
    image_np = image_np.astype(np.float32) / 255.0
    image_np = np.expand_dims(image_np, axis=0)
    return torch.from_numpy(image_np)


class CarLinemarksDataset(Dataset):

    def __init__(self, data_set_folder):
        self.images = glob.glob(data_set_folder + "/*/dataset/*.jpg")
        # TODO : rescale the image
        self.transform = transforms.Compose(
            [transforms.Grayscale(),
             transforms.ToTensor(),
             transforms.Resize((320, 256), antialias=True)])
        print("loaded", len(self.images), "data")
        self.num_labels = len(LABEL_TO_ID)

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


    def __getitem__(self, idx):
        image = self.get_image(idx)
        label = self.get_label(idx)
        # sample = {'image': self.get_image(idx), 'label': self.get_label(idx)}
        # print(sample['image'].shape, )
        return (image, label)
