import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import glob
import cv2
import random
import numpy as np
from dataset.dataset import transfrom_image_np


data_set_folder = "/home/yeliu/Development/LidarMapping/data/map/"
quantize_images_folder = "data/quantize"
quantize_images_file = "data/quantize.txt"
max_num_quantize_image = 200

images_quantize = glob.glob(data_set_folder + "/*/dataset/*.jpg")
indices = list(range(len(images_quantize)))
random.seed(15612)
random.shuffle(indices)

f = open(quantize_images_file, "w")
for i in range(min(max_num_quantize_image, len(indices))):
    image_cv = cv2.imread(images_quantize[i])
    image = transfrom_image_np(image_cv).detach().numpy()

    path_np = quantize_images_folder + "/" + str(i) + ".npy"
    image.tofile(path_np)
    f.write(path_np)
    f.write("\n")
f.close()
