import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import yaml
import onnx
import onnxsim
import torch

import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler
import torchvision.transforms as transforms

from nano_plus import NanoLines, save_model, load_model_weight
from dataset.dataset import CarLinemarksDataset, LABEL_TO_ID, ID_TO_LABEL


config_file = "nano/config/gray_config.yaml"
model_path = "models/model_nano_lines.ckpt"
output_path = "models/model_nano_lines.onnx"

if __name__ == "__main__":
    with open(config_file) as stream:
        try:
            nano_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    det_model = NanoLines(nano_config)
    det_model.load_state_dict(torch.load(model_path))

    dummy_input = torch.randn(1, 1, 320, 256, requires_grad=True)

    torch.onnx.export(
        det_model,
        dummy_input,
        output_path,
        verbose=True,
        keep_initializers_as_inputs=True,
        opset_version=11,
        # export_params=True,        # store the trained parameter weights inside the model file
        # opset_version=10,          # the ONNX version to export the model to
        # do_constant_folding=True,  # whether to execute constant folding for optimization
    )
    print("finished exporting onnx ")

    print("start simplifying onnx ")
    input_data = dummy_input.detach().cpu().numpy()
    model_sim, flag = onnxsim.simplify(output_path, input_data=input_data)
    if flag:
        onnx.save(model_sim, output_path)
        print("simplify onnx successfully")
    else:
        print("simplify onnx failed")

    print("Model saved to:", output_path)
