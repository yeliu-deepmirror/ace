
import math
import re
import sys
import os
import glob
import random
import cv2
import numpy as np

import matplotlib.pyplot as plt
from typing import List, Dict, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchsummary import summary

from read_write_model import read_model, get_intrinsics_matrix, fundamental_21, color_map_value

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ace_network import Encoder


def get_points3d_overlap(points1: Set, points2: Set) -> float:
    return len(points1 & points2) / min(len(points1), len(points2))


def read_image_cv(image_meta, images_path):
    image_path = os.path.join(images_path, image_meta.name)
    return cv2.imread(image_path)


def read_image(image_meta, images_path, resize_ratio):
    from skimage import io
    image_path = os.path.join(images_path, image_meta.name)
    image = io.imread(image_path)
    target_height = int(image.shape[0] * resize_ratio)

    image = TF.to_pil_image(image)
    image = TF.resize(image, target_height)
    return image


def create_encoder(model_path="./ace_encoder_pretrained.pt", feature_dim=512):
    encoder = Encoder(out_channels=feature_dim)
    encoder_state_dict = torch.load(model_path, map_location="cpu")
    num_encoder_features = encoder_state_dict['res2_conv3.weight'].shape[0]
    encoder.load_state_dict(encoder_state_dict)
    encoder.cuda()
    return encoder, num_encoder_features


def check_encoder_match(image_1_meta, image_2_meta, cameras, images_path):
    # read images
    resize_ratio = 1.0

    # extract features from both image
    encoder, num_encoder_features = create_encoder()

    # summary the model encoder
    summary(encoder, (1, 480, 640))

    image_transform = transforms.Compose([
        transforms.Grayscale(), transforms.ToTensor(),
        transforms.Normalize(mean=[0.4], std=[0.25])])
    def extract_feature(image_meta):
        image = read_image(image_meta, images_path, resize_ratio)
        image_torch = image_transform(image).cuda()
        return encoder(image_torch).permute(1, 2, 0)

    features_1 = extract_feature(image_1_meta)
    features_2 = extract_feature(image_2_meta)

    # compute fundamental matrix
    f_21 = fundamental_21(image_1_meta, cameras[image_1_meta.camera_id], image_2_meta, cameras[image_2_meta.camera_id])

    def get_y_by_epiline(line, x):
        return -(line[2] + line[0] * x) / line[1]

    def get_x_by_epiline(line, y):
        return - (y * line[1] + line[2]) / line[0]


    # for each feature find the best match in the target image
    size_factor = 8.0 / resize_ratio
    sample_max_depth = 100.0

    test_border = 10
    test_interval = 10
    for y_raw in np.arange(test_border, features_1.shape[0] - test_border, test_interval):
        for x_raw in np.arange(test_border, features_1.shape[1] - test_border, test_interval):
            pt_1_feature = features_1[y_raw][x_raw]
            x = size_factor * (x_raw + 0.5)
            y = size_factor * (y_raw + 0.5)
            pt_1 = np.array([x, y]).astype(int)
            line = cv2.computeCorrespondEpilines(pt_1.reshape(-1,1,2), 1, f_21)[0][0]

            # check for pixels along the epiline, to find best match

            results = []
            min_length = 20.0
            max_length = 0.0
            for x_target_raw in range(features_2.shape[1]):
                x_target = size_factor * (x_target_raw + 0.5)
                y_target = get_y_by_epiline(line, x_target)
                y_target_raw = int(y_target / size_factor)

                if y_target_raw < 0 or y_target_raw >= features_2.shape[0] - 1:
                    continue

                pt_2_feature = features_2[y_target_raw][x_target_raw]
                feature_distance = (pt_1_feature - pt_2_feature).norm().cpu().detach().numpy()
                results.append([x_target, y_target, feature_distance])
                if feature_distance > max_length:
                    max_length = feature_distance
                if feature_distance < min_length:
                    min_length = feature_distance
            print(x_raw, y_raw, ":", max_length, min_length)

            # render the images
            image_1_cv = read_image_cv(image_1_meta, images_path)
            image_2_cv = read_image_cv(image_2_meta, images_path)

            cv2.circle(image_1_cv, (int(x), int(y)), 15, (0, 255, 255), -1)

            x0, y0 = map(int, [0, get_y_by_epiline(line, 0)])
            x1, y1 = map(int, [image_1_cv.shape[1], get_y_by_epiline(line, image_1_cv.shape[1])])
            cv2.line(image_2_cv, (x0,y0), (x1,y1), (0, 255, 0), 1)

            # draw the match result
            for res in results:
                cv2.circle(image_2_cv, (int(res[0]), int(res[1])), 12, color_map_value(res[2], min_length, max_length), -1)

            image_show = cv2.hconcat([image_1_cv, image_2_cv])
            image_show = cv2.resize(image_show, (int(image_show.shape[1] * 0.5), int(image_show.shape[0] * 0.5)))
            cv2.imwrite("./output/" + str(x_raw) + "_" + str(y_raw) + ".jpg", image_show)
    print("Done the two image test")


if __name__ == '__main__':
    # read camera poses from colmap sparse model
    # get data folder path
    session_path = "./data/20230817T172928+0800_yvr002_car1/colmap"
    colmap_model_path = session_path + "/sparse/1"
    images_path = session_path + "/images"
    print("Read colmap model from " + colmap_model_path)
    cameras, images, points3D = read_model(colmap_model_path)
    print("   # images :", len(images))
    print("   # cameras :", len(cameras))
    print("   # points3D :", len(points3D))

    # pick two images and check the match
    image_key, image = random.choice(list(images.items()))
    image_point3d_ids_set = set(image.point3D_ids.tolist())
    print(" pick image ", image_key)

    max_overlay = 0.0
    max_image_key_target = image_key
    for offset in range(10):
        image_key_target = image_key + offset + 1
        if image_key_target not in images:
            continue
        target_point3d_ids_set = set(images[image_key_target].point3D_ids.tolist())
        overlay = get_points3d_overlap(image_point3d_ids_set, target_point3d_ids_set)
        if overlay > 0.5:  # skip too similiar images
            continue
        if overlay > max_overlay :
            max_image_key_target = image_key_target
            max_overlay = overlay

    print("Test ACE encoder matcher", image_key, "VS", max_image_key_target, ", with overlay", max_overlay)
    check_encoder_match(images[image_key], images[max_image_key_target], cameras, images_path)

    print("Done")
