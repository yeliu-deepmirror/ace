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
data_set_folder = "/home/yeliu/Development/LidarMapping/data/map/"
model_path = "models/model_nano_lines.ckpt"
model_best_path = "models/model_nano_lines_best.ckpt"
train_image = "models/train.png"
max_num_epoches = 3000
use_direction_loss = True
use_point_to_line_loss = False
continue_trainning = False

def get_optimizer(model, opt_cfg):
    cfg_args = opt_cfg.copy()
    func_args = {}
    func_args.update(cfg_args)
    params_opt = filter(lambda p: p.requires_grad, model.parameters())
    return torch.optim.AdamW(params=params_opt, **func_args)


def get_scheduler(optimizer, scheduler_cfg):
    cfg_args = scheduler_cfg.copy()
    func_args = {}
    func_args.update(cfg_args)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, **func_args)


def tmp_dot(a, b):
    return torch.mul(a[:, :, 0], b[:,:,0]) + torch.mul(a[:, :, 1], b[:,:,1])


def tmp_projection(delta, direction):
    dot_prod = tmp_dot(delta, direction)
    return direction * dot_prod[:, :, None]


def label_loss(output, target):
    # first element indicate if the line exist
    # other elements are the line position
    diff_positions = torch.norm(output[:,:,1:] - target[:,:,1:], dim=2)
    loss_positions = torch.norm(torch.mul(diff_positions, target[:,:,0]), dim=1)
    loss_labels = torch.norm(output[:,:,0] - target[:,:,0], dim=1)

    label_weight = 10.0
    loss = loss_positions + label_weight * loss_labels

    direction_gt = torch.nn.functional.normalize(target[:,:,3:5] - target[:,:,1:3], dim=2)
    if use_direction_loss:
        direction_output = torch.nn.functional.normalize(output[:,:,3:5] - output[:,:,1:3], dim=2)
        loss_direction_vec = torch.norm(direction_gt - direction_output, dim=2)
        loss_direction = torch.norm(torch.mul(loss_direction_vec, target[:,:,0]), dim=1)

        # dirction is more important than the position
        # since our position could have large error, during data making
        direction_weight = 2.0
        loss += direction_weight * loss_direction

    if use_point_to_line_loss:
        # add point to line distance loss
        delta_1 = output[:,:,1:3] - target[:,:,1:3]
        loss_projection_vec_1 = torch.norm(delta_1 - tmp_projection(delta_1, direction_gt), dim=2)
        loss_projection_1 = torch.norm(torch.mul(loss_projection_vec_1, target[:,:,0]), dim=1)

        delta_2 = output[:,:,3:5] - target[:,:,1:3]
        loss_projection_vec_2 = torch.norm(delta_2 - tmp_projection(delta_2, direction_gt), dim=2)
        loss_projection_2 = torch.norm(torch.mul(loss_projection_vec_2, target[:,:,0]), dim=1)

        projection_weight = 2.0
        loss += projection_weight * (loss_projection_1 + loss_projection_2)

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
if continue_trainning:
    det_model.load_state_dict(torch.load(model_path))
    # load_model_weight(det_model, model_path)

optimizer = get_optimizer(det_model, nano_config["optimizer_cfg"])
scheduler = None
scheduler = get_scheduler(optimizer, nano_config["lr_scheduler_cfg"])

train_losses = []
test_epoches = []
test_losses = []
best_test_loss = float('inf')
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

    train_losses.append(train_loss_sum)

    if scheduler is not None:
        scheduler.step()

    if epoch%10 == 0:
        with torch.no_grad():
            test_loss_sum = run_test(det_model, test_loader)
            test_epoches.append(epoch)
            test_losses.append(test_loss_sum)
            print(" => test_loss_sum :", test_loss_sum)

            if test_loss_sum < best_test_loss:
                best_test_loss = test_loss_sum
                torch.save(det_model.state_dict(), model_best_path)

    if epoch%20 == 0:
        torch.save(det_model.state_dict(), model_path)
        # save_model(det_model, model_path, epoch, 0)

        plt.clf()
        plt.plot(np.log(np.array(train_losses)), label="train")
        plt.plot(test_epoches, np.log(np.array(test_losses)), label="test")
        plt.legend()
        plt.savefig(train_image)

    print(" => train_loss_sum :", train_loss_sum)
