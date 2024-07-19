import glob
import random
from PIL import Image
import numpy as np

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler
import torchvision.transforms as transforms

class CarLinemarksDataset(Dataset):

    def __init__(self, data_set_folder):
        self.images = glob.glob(data_set_folder + "/*.jpg")
        # TODO : rescale the image
        self.transform = transforms.Compose(
            [transforms.Grayscale(),
             transforms.ToTensor(),
             transforms.Resize((320, 256), antialias=True)])
        print("loaded", len(self.images), "data")
        self.label_to_id = {
            "right": 0,
            "left": 1,
            "bot": 2,
            "chair_bot": 3,
            "front": 4,
            "center": 5,
            "right_chair": 6,
            "left_chair": 7,
        }
        self.num_labels = len(self.label_to_id)

    def __len__(self):
        return len(self.images)

    def get_image(self, idx):
        image = Image.open(self.images[idx])
        image_tensor = self.transform(image)
        return image_tensor

    def get_label(self, idx):
        labels = np.zeros((8, 5))
        label_file = self.images[idx][:-4] + ".txt"
        with open(label_file, 'r') as file_in:
            for line in file_in:
                message = line.split(',')
                idx = self.label_to_id[message[4][:-1]]
                labels[idx, 0] = 1.0
                for i in range(4):
                    labels[idx, i + 1] = float(message[i])
        return labels


    def __getitem__(self, idx):
        image = self.get_image(idx)
        label = self.get_label(idx)
        # sample = {'image': self.get_image(idx), 'label': self.get_label(idx)}
        # print(sample['image'].shape, )
        return (image, label)
