import torch
import glob
import yaml
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
model_path = "models/model_nano_lines.ckpt"
max_num_epoches = 2000

def get_optimizer(model, opt_cfg):
    cfg_args = opt_cfg.copy()
    func_args = {}
    func_args.update(cfg_args)
    return torch.optim.SGD([
        {"params": model.head.parameters()},
        {"params": model.fpn.parameters()},
    ], **func_args)


def get_scheduler(optimizer, scheduler_cfg):
    cfg_args = scheduler_cfg.copy()
    func_args = {}
    func_args.update(cfg_args)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, **func_args)


def label_loss(output, target):
    # first element indicate if the line exist
    # other elements are the line position
    # TODO: make a better loss
    diff_positions = torch.norm(output[:,:,1:] - target[:,:,1:], dim=2)
    loss_positions = torch.norm(torch.mul(diff_positions, target[:,:,0]), dim=1)
    label_weight = 5.0;
    loss_labels = torch.norm(output[:,:,0] - target[:,:,0], dim=1)
    loss = loss_positions + label_weight * loss_labels
    return loss.sum()


def run_test(det_model, test_loader):
    test_loss = 0
    # det_model.cpu()
    for i, data in enumerate(test_loader, 0):
        outputs = det_model(data[0].cuda())
        test_loss += label_loss(outputs.cpu(), data[1]).item()
    return test_loss


with open(config_file) as stream:
    try:
        nano_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
dataset = CarLinemarksDataset(data_set_folder)

# splitting training and testing sets
batch_size = 32
indices = list(range(len(dataset)))
random.seed(15612)
random.shuffle(indices)
split_point = int(0.85*len(indices))
train_indices = indices[:split_point]
test_indices = indices[split_point:]
print("Size of the training set:", len(train_indices))
print("Size of the  testing set:", len(test_indices))

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices))

det_model = NanoLines(nano_config).cuda()

# load the network
# det_model.load_state_dict(torch.load(model_path))
# load_model_weight(det_model, model_path)

optimizer = get_optimizer(det_model, nano_config["optimizer_cfg"])
scheduler = get_scheduler(optimizer, nano_config["lr_scheduler_cfg"])


for epoch in range(max_num_epoches + 1):
    det_model.cuda()
    print("run epoch", epoch)
    train_loss_sum = 0
    for i, data in enumerate(test_loader, 0):
        # Clear gradients
        optimizer.zero_grad()
        outputs = det_model(data[0].cuda())
        loss = label_loss(outputs.cpu(), data[1])
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
    torch.cuda.empty_cache()

    if epoch%10 == 0:
        scheduler.step()
        with torch.no_grad():
            test_loss_sum = run_test(det_model, test_loader)
            print(" => test_loss_sum :", test_loss_sum)
    if epoch%50 == 0:
        torch.save(det_model.state_dict(), model_path)
        # save_model(det_model, model_path, epoch, 0)
    print(" => train_loss_sum :", train_loss_sum)
