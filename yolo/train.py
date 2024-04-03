
import numpy as np
import torch
from yolo_carloc import YoloCarLoc
from torchsummary import summary

# https://github.com/ultralytics/ultralytics
model_yolo = YoloCarLoc("models/yolo/yolov8n.pt")

# get yolo backbone and export
# model_backbone = model_yolo.model.model[0:10]


# model_yolo = YOLO("yolo.yaml").cuda()
# model_yolo.export(format='torchscript')  # creates 'yolov8n.torchscript'
# model_yolo.info(True)
#
#
# script_module = torch.jit.load("yolo.torchscript")
# # summary the model encoder
# x = torch.randn(1, 3, 640, 640).cuda()
# print(model_yolo(x).shape)

summary(model_yolo, (3, 640, 640))
# print(script_module)


# params = sum([np.prod(p.size()) for p in model_yolo.parameters()])
# # print(model_yolo)
# print(params)


# Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# path = model.export(format="onnx")  # export the model to ONNX format
