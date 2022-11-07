from collections import OrderedDict
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.model_selection import train_test_split
import math
import torchvision
from collections import defaultdict
from PIL import Image
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
import torchvision.datasets as datasets
from torch.utils.data import RandomSampler
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import time
import more_itertools as mit
import sys
import copy
import csv
from utilities import *
from exp_environments import *


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

def Clipping_Median(old_weights, new_weights):
    max_norm=20
    grads=[]
    for new_weight in new_weights:
        norm_diff=np.linalg.norm(weights_to_vector(old_weights)-weights_to_vector(new_weight))
        clipped_grad = [(layer_old_weight-layer_new_weight)*min(1,max_norm/norm_diff) for layer_old_weight,layer_new_weight in zip(old_weights, new_weight)]
        grads.append(clipped_grad)

    med_grad=[]
    for layer in range(len(grads[0])):
        lis=[]
        for weight in grads:
            lis.append(weight[layer])
        arr=np.array(lis)
        med_grad.append(np.median(arr,axis=0))

    Centered_weights=[w1-w2 for w1,w2 in zip(old_weights, med_grad)]


    return Centered_weights

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128
q = 0.1 # i.i.d level, 0.1 means i.i.d, higher means non-i.i.d
num_clients = 100
subsample_rate = 0.1
num_attacker = 20
num_class = 10 #For MNIST, Fashion-MNIST and Cifar-10
fl_epoch=200
lr=0.01
num_dummy_batch=1
dummy_batch_size=32

random.seed(150)
att_ids=random.sample(range(num_clients),num_attacker)
att_ids=list(np.sort(att_ids, axis = None))
print('attacker ids: ', att_ids)
setup = dict(device=DEVICE, dtype=torch.float)
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset = datasets.CIFAR10(root='./data',train=False, download=True, transform=transforms.ToTensor())



net = torch.load('cifarinit')
aggregate_weights = get_parameters(net)
old_weights = aggregate_weights

check_index = [i for i in range(5000)]

valset = Subset(trainset, dummy_ids)
att_trainset = Subset(trainset, check_index)

env = FL_cifar_clipping_median(att_trainset, valset)
#env = FL_cifar_krum_accent_large_distribution(att_trainset, valset)


n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="cifar_clipping_median_q_0.1_20att_0.1sample_norm20_2/",name_prefix='rl_model')
model = TD3("MultiInputPolicy", env, buffer_size = 100000,
            policy_kwargs={"net_arch" : [256,128]}, tensorboard_log="cifar_clipping_median_q_0.1_20att_0.1sample_norm20_2/",
            verbose=1, gamma = 1, action_noise = action_noise, learning_rate=1e-7, train_freq = (5, "step"), batch_size = 256)

model.learn(total_timesteps=80000, callback = checkpoint_callback)
