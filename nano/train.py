import torch
import glob
import yaml
from PIL import Image
import matplotlib.pyplot as plt

import random
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler
import torchvision.transforms as transforms

from nano_plus import NanoLines
from dataset.dataset import CarLinemarksDataset

config_file = "nano/config/gray_config.yaml"
data_set_folder = "data/20240222T101812+0800_oppoma_/dataset"


def get_optimizer(opt_cfg):
    cfg_args = opt_cfg.copy()
    func_args = {}
    func_args.update(cfg_args)
    return torch.optim.AdamW(**func_args)


def get_scheduler(scheduler_cfg):
    cfg_args = scheduler_cfg.copy()
    func_args = {}
    func_args.update(cfg_args)
    return torch.optim.lr_scheduler.CosineAnnealingLR(**func_args)


def label_loss(output, target):
    # first element indicate if the line exist
    # other elements are the line position
    # TODO: make a better loss
    loss_positions = target[::,1:]



with open(config_file) as stream:
    try:
        nano_config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
dataset = CarLinemarksDataset(data_set_folder)

# splitting training and testing sets
batch_size = 32
indices = list(range(len(dataset)))
random.shuffle(indices)
split_point = int(0.85*len(indices))
train_indices = indices[:split_point]
test_indices = indices[split_point:]
print("Size of the training set:", len(train_indices))
print("Size of the  testing set:", len(test_indices))

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices))



det_model = NanoLines(nano_config)

# test = det_model(dataset[0][0])
# print(test)

# print(det_model)

for i, data in enumerate(train_loader, 0):
    print(i)
    print(data[0].shape)
    print(data[1].shape)
    test = det_model(data[0])
    print(test)
    break


        # self.optimizer = self.context.wrap_optimizer(
        #     build_optimizer(self.cfg.optimizer_cfg,
        #                     params=filter(lambda p: p.requires_grad, self.model.parameters())))
        # self.optimizer.zero_grad()
        # lr_scheduler = build_lr_scheduler(self.cfg.lr_scheduler_cfg, optimizer=self.optimizer)
        # self.scheduler = self.context.wrap_lr_scheduler(
        #     lr_scheduler,
        #     step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH,
        #     frequency=self.get_lr_scheduler_frequency())

# optimizer = get_optimizer(nano_config["optimizer_cfg"]);
# scheduler = get_scheduler(nano_config["lr_scheduler_cfg"]);
