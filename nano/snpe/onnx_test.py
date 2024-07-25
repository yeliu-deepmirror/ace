import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import onnx
import cv2
import glob
import onnxruntime as ort
import numpy as np
from dataset.dataset import transfrom_image_np

onnx_model_path = "models/model_nano_lines.onnx"
images_folder = "data/car_line_yvr"


onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)

ort_sess = ort.InferenceSession(onnx_model_path)

images = glob.glob(images_folder + "/*.jpg")
for idx in range(len(images)):
    image_cv = cv2.imread(images[idx])
    image = transfrom_image_np(image_cv).unsqueeze(0).detach().numpy()

    outputs = ort_sess.run(None, {'input.1': image})
    print(outputs)
