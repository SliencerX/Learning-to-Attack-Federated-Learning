from collections import OrderedDict
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.decomposition import TruncatedSVD
import math

import torchvision
from collections import defaultdict
from PIL import Image

import os
import random
import torch
#torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.datasets as datasets
from torchvision import transforms, utils

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
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
setup = dict(device=DEVICE, dtype=torch.float)

from stable_baselines3 import DDPG, TD3, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Callable
class FL_mnist_clipping_median(gym.Env):

    def __init__(self, att_trainset, validset):

        self.rnd=0
        self.weights_dimension=1290  #510, 1290, 21840

        high = 1
        low = -high
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(2,),
            dtype=np.float32
        )

        high = np.inf
        low = -high
        self.observation_space = spaces.Dict(pram = spaces.Box(low = -np.inf, high = np.inf, shape = (1290,), dtype = np.float32),
                                             num_attacker = spaces.Discrete(11))
        self.seed()
        self.lr = 0.01
        cc = torch.cat([att_trainset[i][0].reshape(-1) for i in range(len(att_trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
        transform = transforms.Compose([
            transforms.Normalize(data_mean, data_std)])
        att_trainset.transform = transform
        self.att_trainset = att_trainset
        self.valiloader = DataLoader(validset, batch_size = len(validset))
        self.epoch = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.rnd+=1
        action[0] = action[0]*14.9+15 #epsilon [0,10]
        action[1] = action[1]*24+25  #local step [1:1:50]
        new_weights=[]
        for cid in exclude(self.cids,att_ids):
            set_parameters(self.net,self.aggregate_weights)
            train(self.net, self.train_iter, epochs=1, lr=self.lr)
            #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
            new_weight=get_parameters(self.net)
            new_weights.append(new_weight)

        att_weights_lis=[]
        set_parameters(self.net, self.aggregate_weights)
        train(self.net, self.all_train_iter, epochs=int(action[1]), lr=0.01, mode = True)
        new_weight=get_parameters(self.net)
        loss, acc = test(self.net, self.valiloader)
        #print(self.rnd, loss, acc)
        check = int(action[1])
        while np.isnan(loss):
            check = check - 1
            set_parameters(self.net, self.aggregate_weights)
            train(self.net, self.all_train_iter, epochs=check, lr=0.01, mode = True)
            new_weight=get_parameters(self.net)
            loss, acc = test(self.net, self.valiloader)
            print(self.rnd, loss, acc)
        att_weights_lis.append(new_weight)

        for cid in common(self.cids,att_ids):
            new_weight=craft_att(self.aggregate_weights, average(att_weights_lis), -1, action[0])
            new_weights.append(new_weight)
        #print(len(new_weights))

        # Compute average weights
        #self.aggregate_weights =  average(new_weights)
        #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
        self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
        #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
        #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
        #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)

        random.seed(self.rnd)
        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))

        #is_attack=check_attack(self.cids, att_ids)
        is_attack=True
        while is_attack==False:

            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            #print('chosen clients: ', cids)
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            for cid in self.cids:
                set_parameters(self.net,self.aggregate_weights)
                #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))

            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)

        set_parameters(self.net,self.aggregate_weights)
        new_loss, new_acc = test(self.net, self.valiloader)
        reward= new_loss - self.loss
        self.loss=new_loss

        last_layer= np.concatenate([self.aggregate_weights[-2].flatten(),self.aggregate_weights[-1]]).reshape(1,self.weights_dimension)
        state_min = np.min(last_layer)
        state_max = np.max(last_layer)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in last_layer]
        norm_state = np.array(norm_state).reshape(1,1290)
        done=False
        if self.rnd>=1000:
            done= True #15, 25, 75
            _, acc = test(self.net, testloader)
            print(self.rnd, new_loss, acc)
        return {"pram": norm_state, "num_attacker": len(common(self.cids, att_ids))}, reward, done, {}

    def reset(self):
        self.epoch += 1
        if self.epoch > 1:
            del self.train_iter, self.all_train_iter, self.attloader, self.real_att_set
        sample_index = [ i for i in range(min(int((self.epoch-1) * 32 * 2.5 + 200),  len(self.att_trainset)))]
        self.real_att_set = Subset(self.att_trainset, sample_index)
        self.trainloader = DataLoader(self.real_att_set, batch_size = 200, shuffle=True)
        self.attloader = DataLoader(self.real_att_set, batch_size = batch_size, shuffle = True)
        self.train_iter=mit.seekable(self.attloader)
        self.all_train_iter = mit.seekable(self.trainloader)
        self.rnd = 0
        # Load model
        self.net = torch.load('mnist_init').to(**setup)
        self.aggregate_weights = get_parameters(self.net)

        random.seed(self.rnd)
        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
        starts_step=0
        is_attack=True
        while is_attack==False or starts_step>0:
            starts_step-=1
            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            for cid in self.cids:
                set_parameters(self.net,self.aggregate_weights)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))

            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)
        #self.state = weights_to_vector(self.aggregate_weights)
        last_layer= np.concatenate([self.aggregate_weights[-2].flatten(),self.aggregate_weights[-1]]).reshape(1,self.weights_dimension)
        state_min = np.min(last_layer)
        state_max = np.max(last_layer)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in last_layer]
        norm_state = np.array(norm_state).reshape(1,1290)
        set_parameters(self.net,self.aggregate_weights)
        self.loss, self.acc = test(self.net, self.valiloader)
        return {"pram": norm_state, "num_attacker": len(common(self.cids, att_ids))}


class FL_mnist_krum(gym.Env):

    def __init__(self, att_trainset, validset):

        self.rnd=0
        self.weights_dimension=1290  #510, 1290, 21840

        high = 1
        low = -high
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(2,),
            dtype=np.float32
        )
        high = np.inf
        low = -high
        self.observation_space = spaces.Dict(pram = spaces.Box(low = -np.inf, high = np.inf, shape = (1290,), dtype = np.float32),
                                             num_attacker = spaces.Discrete(11))
        self.seed()
        self.lr = 0.01
        cc = torch.cat([att_trainset[i][0].reshape(-1) for i in range(len(att_trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
        transform = transforms.Compose([
            transforms.Normalize(data_mean, data_std)])
        att_trainset.transform = transform
        self.att_trainset = att_trainset
        self.valiloader = DataLoader(validset, batch_size = len(validset))
        self.epoch = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.rnd+=1
        action[0] = action[0]*4.9+5.0 #epsilon [0,10]
        action[1] = action[1]*10+11  #local step [1:1:50]
        new_weights=[]
        for cid in exclude(self.cids,att_ids):
            set_parameters(self.net,self.aggregate_weights)
            train(self.net, self.train_iter, epochs=1, lr=self.lr)
            new_weight=get_parameters(self.net)
            new_weights.append(new_weight)

        att_weights_lis=[]
        set_parameters(self.net, self.aggregate_weights)
        train(self.net, self.all_train_iter, epochs=int(action[1]), lr=0.01, mode = True)
        new_weight=get_parameters(self.net)
        loss, acc = test(self.net, self.valiloader)
        #print(self.rnd, loss, acc)
        check = int(action[1])
        while np.isnan(loss):
            check = check - 1
            set_parameters(net, self.aggregate_weights)
            train(self.net, self.all_train_iter, epochs=check, lr=0.01, mode = True)
            new_weight=get_parameters(net)
            loss, acc = test(net, valiloader)
            print(self.rnd, loss, acc)
        att_weights_lis.append(new_weight)

        for cid in common(self.cids,att_ids):
            new_weight=craft_att(self.aggregate_weights, average(att_weights_lis), -1, action[0])
            new_weights.append(new_weight)

        # Compute average weights
        #self.aggregate_weights =  average(new_weights)
        #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
        #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
        self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
        #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
        #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)

        random.seed(self.rnd)
        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))

        #is_attack=check_attack(self.cids, att_ids)
        is_attack=True
        while is_attack==False:

            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            #print('chosen clients: ', cids)
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            for cid in self.cids:
                set_parameters(self.net,self.aggregate_weights)
                #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))

            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
            self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)

        set_parameters(self.net,self.aggregate_weights)
        #new_loss, _ = test(self.net, simu_testloader)
        new_loss, new_acc = test(self.net, self.valiloader)
        reward= new_loss - self.loss

        self.loss=new_loss

        #self.state = weights_to_vector(self.aggregate_weights)

        last_layer= np.concatenate([self.aggregate_weights[-2].flatten(),self.aggregate_weights[-1]]).reshape(1,self.weights_dimension)
        state_min = np.min(last_layer)
        state_max = np.max(last_layer)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in last_layer]
        norm_state = np.array(norm_state).reshape(1,1290)
        #print(self.state)
        #self.state.append(len(common(self.cids, att_ids)))

        done=False
        #print(self.rnd)
        if self.rnd>=1000:
            done= True #15, 25, 75
            _, acc = test(self.net, testloader)
            print(self.rnd, new_loss, acc)
        #return {"loss": self.loss, "num_selected_attacker": len(common(self.cids, att_ids))-1}, reward, done, {}
        return {"pram": norm_state, "num_attacker": len(common(self.cids, att_ids))}, reward, done, {}

    def reset(self):
        self.epoch += 1
        if self.epoch > 1:
            del self.train_iter, self.all_train_iter, self.attloader, self.real_att_set
        sample_index = [ i for i in range(min(int((self.epoch-1) * 32 *2.5 + 200),  len(self.att_trainset)))]
        self.real_att_set = Subset(self.att_trainset, sample_index)
        self.trainloader = DataLoader(self.real_att_set, batch_size= 200, shuffle=True)
        self.attloader = DataLoader(self.real_att_set, batch_size = batch_size, shuffle = True)
        self.train_iter=mit.seekable(self.attloader)
        self.all_train_iter = mit.seekable(self.trainloader)
        self.rnd = 0
        # Load model
        self.net = torch.load('mnist_init').to(**setup)
        self.aggregate_weights = get_parameters(self.net)
        random.seed(self.rnd)
        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
        # Initial weight start from random step
        starts_step=random.randint(0,1000) #20, 100
        #is_attack=check_attack(self.cids, att_ids)
        starts_step=0
        is_attack=True
        while is_attack==False or starts_step>0:
            starts_step-=1
            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            for cid in self.cids:
                set_parameters(self.net,self.aggregate_weights)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))

            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
            self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)
        #self.state = weights_to_vector(self.aggregate_weights)
        last_layer= np.concatenate([self.aggregate_weights[-2].flatten(),self.aggregate_weights[-1]]).reshape(1,self.weights_dimension)
        state_min = np.min(last_layer)
        state_max = np.max(last_layer)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in last_layer]
        norm_state = np.array(norm_state).reshape(1,1290)
        set_parameters(self.net,self.aggregate_weights)
        self.loss, self.acc = test(self.net, self.valiloader)
        return {"pram": norm_state, "num_attacker": len(common(self.cids, att_ids))}

class FL_emnist_clipping_median(gym.Env):

    def __init__(self, att_trainset, validset):

        self.rnd=0
        self.weights_dimension=6063  #510, 1290, 21840

        high = 1
        low = -high
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(2,),
            dtype=np.float32
        )

        high = np.inf
        low = -high
        self.observation_space = spaces.Dict(pram = spaces.Box(low = -np.inf, high = np.inf, shape = (6063,), dtype = np.float32),
                                             num_attacker = spaces.Discrete(11))
        self.seed()
        self.lr = 0.05
        #_, self.attloader, self.valiloader, testloader, rootloader = load_data_fix_real(num_clients, att_ids, batch_size, 500, seed = 100, bias = 1, rootnumber = 100)
        cc = torch.cat([att_trainset[i][0].reshape(-1) for i in range(len(att_trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
        transform = transforms.Compose([
            transforms.Normalize(data_mean, data_std)])
        att_trainset.transform = transform
        self.att_trainset = att_trainset
        self.valiloader = DataLoader(validset, batch_size = len(validset))
        self.epoch = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.rnd+=1

        action[0] = action[0]*14.9+15 #epsilon [0,10]
        action[1] = action[1]*24+25  #local step [1:1:50]
        new_weights=[]
        for cid in exclude(self.cids,att_ids):
            set_parameters(self.net,self.aggregate_weights)
            train(self.net, self.train_iter, epochs=1, lr=self.lr)
            new_weight=get_parameters(self.net)
            new_weights.append(new_weight)

        att_weights_lis=[]
        set_parameters(self.net, self.aggregate_weights)
        train(self.net, self.all_train_iter, epochs=int(action[1]), lr=0.05, mode = True)
        new_weight=get_parameters(self.net)
        loss, acc = test(self.net, self.valiloader)
        check = int(action[1])
        while np.isnan(loss):
            check = check - 1
            set_parameters(self.net, self.aggregate_weights)
            train(self.net, self.all_train_iter, epochs=check, lr=0.05, mode = True)
            new_weight=get_parameters(self.net)
            loss, acc = test(self.net, self.valiloader)
            print(self.rnd, loss, acc)
        att_weights_lis.append(new_weight)

        for cid in common(self.cids,att_ids):
            new_weight=craft_att(self.aggregate_weights, average(att_weights_lis), -1, action[0])
            new_weights.append(new_weight)

        # Compute average weights
        #self.aggregate_weights =  average(new_weights)
        #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
        self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
        #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
        #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
        #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)

        random.seed(self.rnd)
        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
        is_attack=True
        while is_attack==False:

            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            for cid in self.cids:
                set_parameters(self.net,self.aggregate_weights)
                #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))

            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)

        set_parameters(self.net,self.aggregate_weights)
        new_loss, new_acc = test(self.net, self.valiloader)
        reward= self.acc-new_acc

        self.loss=new_loss
        self.acc = new_acc

        last_layer= np.concatenate([self.aggregate_weights[-2].flatten(),self.aggregate_weights[-1]]).reshape(1,self.weights_dimension)
        state_min = np.min(last_layer)
        state_max = np.max(last_layer)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in last_layer]
        norm_state = np.array(norm_state).reshape(1,6063)
        done=False
        if self.rnd>=1000:
            done= True #15, 25, 75
            _, acc = test(self.net, testloader)
            print(self.rnd, new_loss, acc)
        return {"pram": norm_state, "num_attacker": len(common(self.cids, att_ids))}, reward, done, {}

    def reset(self):
        self.epoch += 1
        if self.epoch > 1:
            del self.train_iter, self.all_train_iter, self.attloader, self.real_att_set
        sample_index = [ i for i in range(min(int((self.epoch-1) * 32 * 2.5 + 500),  len(self.att_trainset)))]
        self.real_att_set = Subset(self.att_trainset, sample_index)
        self.trainloader = DataLoader(self.real_att_set, batch_size = 500, shuffle=True)
        self.attloader = DataLoader(self.real_att_set, batch_size = batch_size, shuffle = True)
        self.train_iter=mit.seekable(self.attloader)
        self.all_train_iter = mit.seekable(self.trainloader)
        self.rnd = 0
        # Load model
        self.net = torch.load('emnist_init').to(**setup)
        self.aggregate_weights = get_parameters(self.net)
        random.seed(self.rnd)
        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
        starts_step=0
        is_attack=True
        while is_attack==False or starts_step>0:
            starts_step-=1
            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            #for i in range(int(num_clients*subsample_rate)):
            for cid in self.cids:
                set_parameters(self.net,self.aggregate_weights)
                #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))

            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)
        #self.state = weights_to_vector(self.aggregate_weights)
        last_layer= np.concatenate([self.aggregate_weights[-2].flatten(),self.aggregate_weights[-1]]).reshape(1,self.weights_dimension)
        state_min = np.min(last_layer)
        state_max = np.max(last_layer)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in last_layer]
        norm_state = np.array(norm_state).reshape(1,6063)
        set_parameters(self.net,self.aggregate_weights)
        self.loss, self.acc = test(self.net, self.valiloader)
        return {"pram": norm_state, "num_attacker": len(common(self.cids, att_ids))}


class FL_emnist_krum(gym.Env):

    def __init__(self, att_trainset, validset):

        self.rnd=0
        self.weights_dimension=6063  #510, 1290, 21840

        high = 1
        low = -high
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(2,),
            dtype=np.float32
        )

        high = np.inf
        low = -high
        self.observation_space = spaces.Dict(pram = spaces.Box(low = -np.inf, high = np.inf, shape = (6063,), dtype = np.float32),
                                             num_attacker = spaces.Discrete(11))
        self.seed()
        self.lr = 0.05
        #_, self.attloader, self.valiloader, testloader, rootloader = load_data_fix_real(num_clients, att_ids, batch_size, 500, seed = 100, bias = 1, rootnumber = 100)
        cc = torch.cat([att_trainset[i][0].reshape(-1) for i in range(len(att_trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)
        transform = transforms.Compose([
            transforms.Normalize(data_mean, data_std)])
        att_trainset.transform = transform
        self.att_trainset = att_trainset
        self.valiloader = DataLoader(validset, batch_size = len(validset))
        self.epoch = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.rnd+=1
        action[0] = action[0]*4.9+5.0 #epsilon [0,10]
        action[1] = action[1]*10+11  #local step [1:1:50]
        new_weights=[]
        for cid in exclude(self.cids,att_ids):
            set_parameters(self.net,self.aggregate_weights)
            train(self.net, self.train_iter, epochs=1, lr=self.lr)
            new_weight=get_parameters(self.net)
            new_weights.append(new_weight)

        att_weights_lis=[]
        set_parameters(self.net, self.aggregate_weights)
        train(self.net, self.all_train_iter, epochs=int(action[1]), lr=0.05, mode = True)
        new_weight=get_parameters(self.net)
        loss, acc = test(self.net, self.valiloader)
        #print(self.rnd, loss, acc)
        check = int(action[1])
        while np.isnan(loss):
            check = check - 1
            set_parameters(net, self.aggregate_weights)
            train(self.net, self.all_train_iter, epochs=check, lr=action[1], mode = True)
            new_weight=get_parameters(net)
            loss, acc = test(net, valiloader)
            print(self.rnd, loss, acc)
        att_weights_lis.append(new_weight)

        for cid in common(self.cids,att_ids):
            new_weight=craft_att(self.aggregate_weights, average(att_weights_lis), -1, action[0])
            new_weights.append(new_weight)
        #print(len(new_weights))

        # Compute average weights
        #self.aggregate_weights =  average(new_weights)
        #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
        #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
        self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
        #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
        #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)

        random.seed(self.rnd)
        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))

        #is_attack=check_attack(self.cids, att_ids)
        is_attack=True
        while is_attack==False:

            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            #print('chosen clients: ', cids)
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            for cid in self.cids:
                set_parameters(self.net,self.aggregate_weights)
                #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))

            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
            self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)

        set_parameters(self.net, self.aggregate_weights)
        new_loss, new_acc = test(self.net, self.valiloader)
        if np.isnan(loss):
          self.old_weights = self.old_weights
        else:
          self.old_weights = self.aggregate_weights
        set_parameters(self.net, self.old_weights)
        new_loss, new_acc = test(self.net, self.valiloader)
        reward= new_loss - self.loss

        self.loss=new_loss

        last_layer= np.concatenate([self.aggregate_weights[-2].flatten(),self.aggregate_weights[-1]]).reshape(1,self.weights_dimension)
        state_min = np.min(last_layer)
        state_max = np.max(last_layer)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in last_layer]
        norm_state = np.array(norm_state).reshape(1,6063)
        done=False
        if self.rnd>=1000:
            done= True #15, 25, 75
            _, acc = test(self.net, testloader)
            print(self.rnd, new_loss, acc)
        return {"pram": norm_state, "num_attacker": len(common(self.cids, att_ids))}, reward, done, {}

    def reset(self):
        self.epoch += 1
        if self.epoch > 1:
            del self.train_iter, self.all_train_iter, self.attloader, self.real_att_set
        sample_index = [ i for i in range(min(int((self.epoch-1) * 32 *2.5 + 200),  len(self.att_trainset)))]
        self.real_att_set = Subset(self.att_trainset, sample_index)
        #self.valiloader = DataLoader(self.real_att_set, batch_size=len(self.real_att_set), shuffle=True)
        self.trainloader = DataLoader(self.real_att_set, batch_size= 200, shuffle=True)
        self.attloader = DataLoader(self.real_att_set, batch_size = batch_size, shuffle = True)
        self.train_iter=mit.seekable(self.attloader)
        self.all_train_iter = mit.seekable(self.trainloader)
        self.rnd = 0
        self.net = torch.load('emnist_init').to(**setup)
        self.aggregate_weights = get_parameters(self.net)
        random.seed(self.rnd)
        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
        starts_step=0
        is_attack=True
        while is_attack==False or starts_step>0:
            starts_step-=1
            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            #for i in range(int(num_clients*subsample_rate)):
            for cid in self.cids:
                set_parameters(self.net,self.aggregate_weights)
                #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))

            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
            self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)
        #self.state = weights_to_vector(self.aggregate_weights)
        last_layer= np.concatenate([self.aggregate_weights[-2].flatten(),self.aggregate_weights[-1]]).reshape(1,self.weights_dimension)
        state_min = np.min(last_layer)
        state_max = np.max(last_layer)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in last_layer]
        norm_state = np.array(norm_state).reshape(1,6063)
        set_parameters(self.net,self.aggregate_weights)
        self.loss, self.acc = test(self.net, self.valiloader)
        return {"pram": norm_state, "num_attacker": len(common(self.cids, att_ids))}



class FL_mnist_fltrust_g_td3_acsend_real_test(gym.Env):

    def __init__(self):
        self.rnd=0
        #self.weights_dimension=1290
        high = 1
        low = -1
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(3,),
            dtype=np.float32
        )
        self.observation_space = spaces.Dict(pram = spaces.Box(low = -np.inf, high = np.inf, shape = (6063,), dtype = np.float32),
                                             num_attacker = spaces.Discrete(11))
        self.seed()
        self.lr = 0.05
        dummy_att_ids = [i for i in range(20)]
        _, self.attloader, self.valiloader, self.testloader, rootloader, att_split, _ = emnist_load_data_fix_real(100, dummy_att_ids, 128, 1000, seed = 100, bias = 1, rootnumber = 100)
        self.root_iter = mit.seekable(rootloader)
        self.train_iter=mit.seekable(self.attloader)
        check_index = [i for i in range(500)]
        att_sub_split = Subset(att_split, check_index)
        self.att_test_loader = DataLoader(att_sub_split, batch_size = 500)
        self.att_test_iter = mit.seekable(self.att_test_loader)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.rnd+=1

        set_parameters(self.net, self.aggregate_weights)

        #action[0] = action[0]*4.9+5.0 #epsilon [0,10]
        action[0] = action[0]*0.04+0.05 #lr for local trianning [0, 0.1]
        action[1] = action[1]*10+11  #local step [1:1:50]
        #action[1] = 2
        alpha = action[-1] *0.5 + 0.5
        #action[0] = 0.1
        #action[1] = 5

        #alpha = 0.5
        new_weights=[]
        for cid in exclude(self.cids,att_ids):
            set_parameters(self.net,self.aggregate_weights)
            train(self.net, self.train_iter, epochs=1, lr=self.lr)
            #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
            new_weight=get_parameters(self.net)
            new_weights.append(new_weight)

        att_weights_lis=[]

        set_parameters(self.net, self.aggregate_weights)
        for step in range(int(action[1])):
            #att_acsend(self.net, self.aggregate_weights, self.g_weight, self.train_iter, self.lr, alpha)
            att_acsend_root(self.net, self.aggregate_weights, self.att_test_iter, self.guess_rootiter, action[0], alpha)
            att_weight = get_parameters(self.net)
            g_grad = [old - new for old, new in zip(self.aggregate_weights, self.g_weight)]
            vec_g_grad = weights_to_vector(g_grad)
            vec_att_grad = (weights_to_vector(self.aggregate_weights) - weights_to_vector(att_weight))/(np.linalg.norm(weights_to_vector(self.aggregate_weights) - weights_to_vector(att_weight)))
            vec_att_true_grad = vec_att_grad * np.linalg.norm(vec_g_grad)
            att_true_grad = vector_to_weights(vec_att_true_grad, self.aggregate_weights)
            att_true_weight = [old - grad for old, grad in zip(self.aggregate_weights, att_true_grad)]
            set_parameters(self.net, att_true_weight)

        loss, acc = test(self.net, self.valiloader)
        #print(self.rnd, loss, acc)
        # check = int(action[1])
        # while np.isnan(loss):
        #     check -= 1
        #     set_parameters(self.net, self.aggregate_weights)
        #     for step in range(check):
        #         att_acsend(self.net, self.aggregate_weights, self.g_weight, self.train_iter, self.lr, alpha)
        #     loss, acc = test(self.net, self.valiloader)
        #     print(self.rnd, loss, acc)

        # att_weight = get_parameters(self.net)
        # vec_g_grad = weights_to_vector(self.g_weight)
        # vec_att_grad = (weights_to_vector(self.aggregate_weights) - weights_to_vector(att_weight))/(np.linalg.norm(weights_to_vector(self.aggregate_weights) - weights_to_vector(att_weight)))
        # vec_att_true_grad = vec_att_grad * np.linalg.norm(vec_g_grad)
        # att_true_grad = vector_to_weights(vec_att_true_grad, self.aggregate_weights)
        # att_true_weight = [old - grad for old, grad in zip(self.aggregate_weights, att_true_grad)]
        # set_parameters(self.net, self.aggregate_weights)
        # train(self.net, self.train_iter, epochs=int(action[2]), lr=action[1], mode = False)
        # new_weight=get_parameters(self.net)
        # att_weights_lis.append(new_weight)
        # loss, acc = test(self.net, self.testloader)
        # print(self.rnd, loss, acc)


        for cid in common(self.cids, att_ids):
          new_weight = copy.deepcopy(att_true_weight)
          new_weights.append(new_weight)

        # Compute average weights
        #self.aggregate_weights =  average(new_weights)
        #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
        #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
        #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
        #print(len(new_weights))
        #print(self.cids)
        #print(len(common(self.cids, att_ids)))
        self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter, self.g_weight, lr = self.lr)
        #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)

        set_parameters(self.net,self.aggregate_weights)
        #loss, acc = test(self.net, self.testloader)
        #print(self.rnd, loss, acc)

        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))

        #is_attack=check_attack(self.cids, att_ids)
        is_attack = True
        while is_attack==False:

            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            #print('chosen clients: ', cids)
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            for i in range(int(num_clients*subsample_rate)):
                set_parameters(self.net,self.aggregate_weights)
                #train(self.net, simu_trainloader, epochs=1, lr=lr)
                #train(net, trainloaders[cid], epochs=1, lr=lr)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))

            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter, lr = self.lr)
            #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)

        set_parameters(self.net,self.aggregate_weights)
        #new_loss, _ = test(self.net, simu_testloader)
        new_loss, acc = test(self.net, self.valiloader)
        #new_loss, _ = test(self.net, testloader)
        # Caculate the reward by l(s^t(tau+1))-l(s^t(tau))
        reward = new_loss-self.loss
        #print(reward)
        #reward = self.total_cv - total_cv
        # self.total_cv = total_cv
        #reward = self.acc - acc
        #reward = -acc
        self.loss=new_loss
        self.acc = acc
        #state = np.concatenate((self.g_weight[-2], self.g_weight[-1]), axis = None)
        state = np.concatenate((self.aggregate_weights[-2], self.aggregate_weights[-1]), axis = None)
        state_min = np.min(state)
        state_max = np.max(state)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in state]
        num_attacker = len(common(self.cids, att_ids))
        set_parameters(self.net, self.aggregate_weights)
        #self.root_iter.seek(0)
        train(self.net, self.root_iter, epochs=1, lr=self.lr)
        self.g_weight=get_parameters(self.net)
        done=False

        if self.rnd>=500:
            done = True
            #print(action[:10])
            _, acc = test(self.net, self.testloader)
            print(self.rnd, new_loss, acc)
        return {"pram": norm_state, "num_attacker": num_attacker}, reward, done, {}

    def reset(self):
        #_, self.valiloader, _, rootloader = load_data_fix(batch_size=batch_size, seed = random.randint(100,10000))
        #_, self.valiloader, _, rootloader = load_data_fix(batch_size=batch_size)
        self.rnd=0
        self.net = torch.load("emnist_init").to(**setup)
        self.aggregate_weights = get_parameters(self.net)
        #self.train_iter = mit.seekable(trainloader)
        self.cids=random.sample(range(100),int(100*0.1))
        _, _, _, _, _, _, guess_rootloader = emnist_load_data_fix_real(100, [1,2], 128, 1000, seed = random.randint(100,10000), bias = 1, rootnumber = 100)
        self.guess_rootiter = mit.seekable(guess_rootloader)
        self.guess_rootiter.seek(0)
        # Initial weight start from random step
        #starts_step=random.randint(0,40) #20, 100
        starts_step=0
        #print(starts_step)
        is_attack=True
        while is_attack==False or starts_step>0:

            starts_step-=1
            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            for i in range(int(num_clients*subsample_rate)):
                set_parameters(self.net,self.aggregate_weights)
                #train(self.net, simu_trainloader, epochs=1, lr=lr)
                #train(net, trainloaders[cid], epochs=1, lr=lr)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))
            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter, lr = self.lr)
        set_parameters(self.net, self.aggregate_weights)
        self.root_iter.seek(0)
        train(self.net, self.root_iter, epochs=1, lr=self.lr)
        self.g_weight=get_parameters(self.net)

        state = np.concatenate((self.aggregate_weights[-2], self.aggregate_weights[-1]), axis = None)
        num_attacker = len(common(self.cids, att_ids))
        state_min = np.min(state)
        state_max = np.max(state)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in state]
        num_attacker = len(common(self.cids, att_ids))
        set_parameters(self.net,self.aggregate_weights)
        #self.loss, _ = test(self.net, simu_testloader)
        self.loss, self.acc = test(self.net, self.valiloader)
        #self.loss, _ = test(self.net, testloader)

        return {"pram": norm_state, "num_attacker": num_attacker}





class FL_mnist_fltrust_g_td3_acsend_real_test(gym.Env):

    def __init__(self):
        self.rnd=0
        #self.weights_dimension=1290
        high = 1
        low = -1
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(3,),
            dtype=np.float32
        )
        self.observation_space = spaces.Dict(pram = spaces.Box(low = -np.inf, high = np.inf, shape = (6063,), dtype = np.float32),
                                             num_attacker = spaces.Discrete(11))
        self.seed()
        self.lr = 0.05
        _, self.attloader, self.valiloader, self.testloader, rootloader, att_split = emnist_load_data_fix_real(num_clients, att_ids, batch_size, 500, seed = 100, bias = 1, rootnumber = 100)
        self.root_iter = mit.seekable(rootloader)
        self.train_iter=mit.seekable(self.attloader)
        check_index = [i for i in range(500)]
        att_sub_split = Subset(att_split, check_index)
        self.att_test_loader = DataLoader(att_sub_split, batch_size = 500)
        self.att_test_iter = mit.seekable(self.att_test_loader)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.rnd+=1

        set_parameters(self.net, self.aggregate_weights)

        #action[0] = action[0]*4.9+5.0 #epsilon [0,10]
        action[0] = action[0]*0.04+0.05 #lr for local trianning [0, 0.1]
        action[1] = action[1]*10+11  #local step [1:1:50]
        #action[1] = 2
        alpha = action[-1] *0.5 + 0.5
        #action[0] = 0.1
        #action[1] = 5

        #alpha = 0.5
        new_weights=[]
        for cid in exclude(self.cids,att_ids):
            set_parameters(self.net,self.aggregate_weights)
            train(self.net, self.train_iter, epochs=1, lr=self.lr)
            #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
            new_weight=get_parameters(self.net)
            new_weights.append(new_weight)

        att_weights_lis=[]

        set_parameters(self.net, self.aggregate_weights)
        for step in range(int(action[1])):
            #att_acsend(self.net, self.aggregate_weights, self.g_weight, self.train_iter, self.lr, alpha)
            att_acsend_root(self.net, self.aggregate_weights, self.g_weight, self.att_test_iter, self.root_iter, action[0], alpha)
            att_weight = get_parameters(self.net)
            g_grad = [old - new for old, new in zip(self.aggregate_weights, self.g_weight)]
            vec_g_grad = weights_to_vector(g_grad)
            vec_att_grad = (weights_to_vector(self.aggregate_weights) - weights_to_vector(att_weight))/(np.linalg.norm(weights_to_vector(self.aggregate_weights) - weights_to_vector(att_weight)))
            vec_att_true_grad = vec_att_grad * np.linalg.norm(vec_g_grad)
            att_true_grad = vector_to_weights(vec_att_true_grad, self.aggregate_weights)
            att_true_weight = [old - grad for old, grad in zip(self.aggregate_weights, att_true_grad)]
            set_parameters(self.net, att_true_weight)

        loss, acc = test(self.net, self.valiloader)
        #print(self.rnd, loss, acc)
        # check = int(action[1])
        # while np.isnan(loss):
        #     check -= 1
        #     set_parameters(self.net, self.aggregate_weights)
        #     for step in range(check):
        #         att_acsend(self.net, self.aggregate_weights, self.g_weight, self.train_iter, self.lr, alpha)
        #     loss, acc = test(self.net, self.valiloader)
        #     print(self.rnd, loss, acc)

        # att_weight = get_parameters(self.net)
        # vec_g_grad = weights_to_vector(self.g_weight)
        # vec_att_grad = (weights_to_vector(self.aggregate_weights) - weights_to_vector(att_weight))/(np.linalg.norm(weights_to_vector(self.aggregate_weights) - weights_to_vector(att_weight)))
        # vec_att_true_grad = vec_att_grad * np.linalg.norm(vec_g_grad)
        # att_true_grad = vector_to_weights(vec_att_true_grad, self.aggregate_weights)
        # att_true_weight = [old - grad for old, grad in zip(self.aggregate_weights, att_true_grad)]
        # set_parameters(self.net, self.aggregate_weights)
        # train(self.net, self.train_iter, epochs=int(action[2]), lr=action[1], mode = False)
        # new_weight=get_parameters(self.net)
        # att_weights_lis.append(new_weight)
        # loss, acc = test(self.net, self.testloader)
        # print(self.rnd, loss, acc)


        for cid in common(self.cids, att_ids):
          new_weight = copy.deepcopy(att_true_weight)
          new_weights.append(new_weight)

        # Compute average weights
        #self.aggregate_weights =  average(new_weights)
        #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
        #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
        #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
        #print(len(new_weights))
        #print(self.cids)
        #print(len(common(self.cids, att_ids)))
        self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter, self.g_weight, lr = self.lr)
        #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)

        set_parameters(self.net,self.aggregate_weights)
        #loss, acc = test(self.net, self.testloader)
        #print(self.rnd, loss, acc)

        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))

        #is_attack=check_attack(self.cids, att_ids)
        is_attack = True
        while is_attack==False:

            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            #print('chosen clients: ', cids)
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            for i in range(int(num_clients*subsample_rate)):
                set_parameters(self.net,self.aggregate_weights)
                #train(self.net, simu_trainloader, epochs=1, lr=lr)
                #train(net, trainloaders[cid], epochs=1, lr=lr)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))

            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter, lr = self.lr)
            #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)

        set_parameters(self.net,self.aggregate_weights)
        #new_loss, _ = test(self.net, simu_testloader)
        new_loss, acc = test(self.net, self.valiloader)
        #new_loss, _ = test(self.net, testloader)
        # Caculate the reward by l(s^t(tau+1))-l(s^t(tau))
        reward = new_loss-self.loss
        #print(reward)
        #reward = self.total_cv - total_cv
        # self.total_cv = total_cv
        #reward = self.acc - acc
        #reward = -acc
        self.loss=new_loss
        self.acc = acc
        #state = np.concatenate((self.g_weight[-2], self.g_weight[-1]), axis = None)
        state = np.concatenate((self.aggregate_weights[-2], self.aggregate_weights[-1]), axis = None)
        state_min = np.min(state)
        state_max = np.max(state)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in state]
        num_attacker = len(common(self.cids, att_ids))
        set_parameters(self.net, self.aggregate_weights)
        #self.root_iter.seek(0)
        train(self.net, self.root_iter, epochs=1, lr=self.lr)
        self.g_weight=get_parameters(self.net)
        done=False

        if self.rnd>=500:
            done = True
            #print(action[:10])
            _, acc = test(self.net, self.testloader)
            print(self.rnd, new_loss, acc)
        return {"pram": norm_state, "num_attacker": num_attacker}, reward, done, {}

    def reset(self):
        #_, self.valiloader, _, rootloader = load_data_fix(batch_size=batch_size, seed = random.randint(100,10000))
        #_, self.valiloader, _, rootloader = load_data_fix(batch_size=batch_size)
        self.rnd=0
        self.net = torch.load("emnist_init").to(**setup)
        self.aggregate_weights = get_parameters(self.net)
        #self.train_iter = mit.seekable(trainloader)
        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
        # Initial weight start from random step
        #starts_step=random.randint(0,40) #20, 100
        starts_step=0
        #print(starts_step)
        is_attack=check_attack(self.cids, att_ids)
        while is_attack==False or starts_step>0:

            starts_step-=1
            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            for i in range(int(num_clients*subsample_rate)):
                set_parameters(self.net,self.aggregate_weights)
                #train(self.net, simu_trainloader, epochs=1, lr=lr)
                #train(net, trainloaders[cid], epochs=1, lr=lr)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))
            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter, lr = self.lr)
        #last_layer= np.concatenate([self.aggregate_weights[-2].flatten(),self.aggregate_weights[-1]]).reshape(1,1290)
        #self.state = pca.transform(last_layer)
        #print(self.state)
        set_parameters(self.net, self.aggregate_weights)
        self.root_iter.seek(0)
        train(self.net, self.root_iter, epochs=1, lr=self.lr)
        self.g_weight=get_parameters(self.net)

        state = np.concatenate((self.aggregate_weights[-2], self.aggregate_weights[-1]), axis = None)
        num_attacker = len(common(self.cids, att_ids))
        state_min = np.min(state)
        state_max = np.max(state)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in state]
        num_attacker = len(common(self.cids, att_ids))
        set_parameters(self.net,self.aggregate_weights)
        #self.loss, _ = test(self.net, simu_testloader)
        self.loss, self.acc = test(self.net, self.valiloader)
        #self.loss, _ = test(self.net, testloader)

        return {"pram": norm_state, "num_attacker": num_attacker}

class FL_cifar_clipping_median(gym.Env):

    def __init__(self, att_trainset, validset):

        self.rnd=0
        self.weights_dimension=5130  #510, 1290, 21840

        high = 1
        low = -high
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(2,),
            dtype=np.float32
        )

        high = np.inf
        low = -high
        self.observation_space = spaces.Dict(pram = spaces.Box(low = -np.inf, high = np.inf, shape = (5130,), dtype = np.float32),
                                             num_attacker = spaces.Discrete(11))

        self.seed()
        self.lr = 0.01
        self.att_trainset = att_trainset
        self.valiloader = DataLoader(validset, batch_size = len(validset))
        self.epoch = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.rnd+=1

        action[0] = action[0]*24+25 #epsilon [0,10]
        action[1] = action[1]*12+13  #local step [1:1:50]
        new_weights=[]
        for cid in exclude(self.cids,att_ids):
            set_parameters(self.net,self.aggregate_weights)
            train(self.net, self.train_iter, epochs=1, lr=self.lr)
            new_weight=get_parameters(self.net)
            new_weights.append(new_weight)


        att_weights_lis=[]
        set_parameters(self.net, self.aggregate_weights)
        train(self.net, self.all_train_iter, epochs=int(action[1]), lr=0.01, mode = True)
        new_weight=get_parameters(self.net)
        loss, acc = test(self.net, self.valiloader)
        check = int(action[1])
        while np.isnan(loss):
            check = check - 1
            set_parameters(self.net, self.aggregate_weights)
            train(self.net, self.all_train_iter, epochs=check, lr=0.01, mode = True)
            new_weight=get_parameters(self.net)
            loss, acc = test(self.net, self.valiloader)
            print(self.rnd, loss, acc)
        att_weights_lis.append(new_weight)

        for cid in common(self.cids,att_ids):
            new_weight=craft_att(self.aggregate_weights, average(att_weights_lis), -1, action[0])
            new_weights.append(new_weight)

        # Compute average weights
        #self.aggregate_weights =  average(new_weights)
        #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
        self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
        #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
        #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
        #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)

        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))

        is_attack=True
        while is_attack==False:

            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            for cid in self.cids:
                set_parameters(self.net,self.aggregate_weights)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))

            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)

        set_parameters(self.net,self.aggregate_weights)
        new_loss, new_acc = test(self.net, self.valiloader)
        reward= new_loss - self.loss

        self.loss=new_loss


        last_layer= np.concatenate([self.aggregate_weights[-2].flatten(),self.aggregate_weights[-1]]).reshape(1,self.weights_dimension)
        state_min = np.min(last_layer)
        state_max = np.max(last_layer)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in last_layer]
        norm_state = np.array(norm_state).reshape(1,5130)

        done=False
        #print(self.rnd)
        if self.rnd>=1000:
            done= True #15, 25, 75
            _, acc = test(self.net, testloader)
            print(self.rnd, new_loss, acc)
        #return {"loss": self.loss, "num_selected_attacker": len(common(self.cids, att_ids))-1}, reward, done, {}
        return {"pram": norm_state, "num_attacker": len(common(self.cids, att_ids))}, reward, done, {}

    def reset(self):
        self.epoch += 1
        if self.epoch > 1:
            del self.train_iter, self.all_train_iter, self.attloader, self.real_att_set
        sample_index = [i for i in range(5000)]
        self.real_att_set = Subset(self.att_trainset, sample_index)
        self.trainloader = DataLoader(self.real_att_set, batch_size = 500, shuffle=True)
        self.attloader = DataLoader(self.real_att_set, batch_size = batch_size, shuffle = True)
        self.train_iter=mit.seekable(self.attloader)
        self.all_train_iter = mit.seekable(self.trainloader)
        self.rnd = 0
        # Load model
        #self.net = Net().to(**setup)
        #self.net.load_state_dict(torch.load('small_net_init'))
        self.net = torch.load('resinit').to(**setup)
        #self.net = MNISTClassifier().to(**setup)
        self.aggregate_weights = get_parameters(self.net)
        #self.train_iter = mit.seekable(trainloader)
        random.seed(self.rnd)
        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
        starts_step=0
        is_attack=True
        while is_attack==False or starts_step>0:
            starts_step-=1
            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            #for i in range(int(num_clients*subsample_rate)):
            for cid in self.cids:
                set_parameters(self.net,self.aggregate_weights)
                #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))

            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)
        #self.state = weights_to_vector(self.aggregate_weights)
        last_layer= np.concatenate([self.aggregate_weights[-2].flatten(),self.aggregate_weights[-1]]).reshape(1,self.weights_dimension)
        state_min = np.min(last_layer)
        state_max = np.max(last_layer)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in last_layer]
        norm_state = np.array(norm_state).reshape(1,5130)
        #print(self.state)
        set_parameters(self.net,self.aggregate_weights)
        self.loss, self.acc = test(self.net, self.valiloader)

        return {"pram": norm_state, "num_attacker": len(common(self.cids, att_ids))}

class FL_cifar_krum(gym.Env):

    def __init__(self, att_trainset, validset):

        self.rnd=0
        self.weights_dimension=5130  #510, 1290, 21840

        high = 1
        low = -high
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(2,),
            dtype=np.float32
        )

        high = np.inf
        low = -high
        self.observation_space = spaces.Dict(pram = spaces.Box(low = -np.inf, high = np.inf, shape = (5130,), dtype = np.float32),
                                             num_attacker = spaces.Discrete(11))
        self.seed()
        self.lr = 0.01
        self.att_trainset = att_trainset
        self.valiloader = DataLoader(validset, batch_size = len(validset))
        self.epoch = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.rnd+=1
        action[0] = action[0]*5+6 #epsilon [0,10]
        action[1] = action[1]*5+6  #local step [1:1:50]
        new_weights=[]
        for cid in exclude(self.cids,att_ids):
            set_parameters(self.net,self.aggregate_weights)
            train(self.net, self.train_iter, epochs=1, lr=self.lr)
            #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
            new_weight=get_parameters(self.net)
            new_weights.append(new_weight)


        att_weights_lis=[]
        set_parameters(self.net, self.aggregate_weights)

        train(self.net, self.all_train_iter, epochs=int(action[1]), lr=0.01, mode = True)

        new_weight=get_parameters(self.net)
        loss, acc = test(self.net, self.valiloader)
        #print(self.rnd, loss, acc)
        check = int(action[1])
        while np.isnan(loss):
            check = check - 1
            set_parameters(self.net, self.aggregate_weights)
            train(self.net, self.all_train_iter, epochs=check, lr=0.01, mode = True)
            new_weight=get_parameters(self.net)
            loss, acc = test(self.net, self.valiloader)
            print(self.rnd, loss, acc)
        att_weights_lis.append(new_weight)

        new_weight=craft_att(self.aggregate_weights, average(att_weights_lis), -1, action[0])
        print(len(common(self.cids,att_ids)))
        for cid in common(self.cids,att_ids):
            new_weights.append(new_weight)
        #print(len(new_weights))
        num_attacker = len(common(self.cids, att_ids))
        # Compute average weights
        #self.aggregate_weights =  average(new_weights)
        #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
        #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
        self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
        #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
        #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)

        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))

        #is_attack=check_attack(self.cids, att_ids)
        is_attack=True
        while is_attack==False:

            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            #print('chosen clients: ', cids)
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            for cid in self.cids:
                set_parameters(self.net,self.aggregate_weights)
                #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))

            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
            self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)

        set_parameters(self.net,self.aggregate_weights)
        new_loss, new_acc = test(self.net, self.valiloader)
        print(self.rnd, new_loss, new_acc)
        reward= new_loss - self.loss

        self.loss=new_loss

        last_layer= np.concatenate([self.aggregate_weights[-2].flatten(),self.aggregate_weights[-1]]).reshape(1,self.weights_dimension)
        state_min = np.min(last_layer)
        state_max = np.max(last_layer)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in last_layer]
        norm_state = np.array(norm_state).reshape(1,5130)

        done=False
        if self.rnd>=1000:
            done= True #15, 25, 75
            _, acc = test(self.net, testloader)
            print(self.rnd, new_loss, acc)
        return {"pram": norm_state, "num_attacker": len(common(self.cids, att_ids))}, reward, done, {}

    def reset(self):
        self.epoch += 1
        if self.epoch > 1:
            del self.train_iter, self.all_train_iter, self.attloader, self.real_att_set
        sample_index = [i for i in range(5000)]
        self.real_att_set = Subset(self.att_trainset, sample_index)
        self.trainloader = DataLoader(self.real_att_set, batch_size = 500, shuffle=True)
        self.attloader = DataLoader(self.real_att_set, batch_size = batch_size, shuffle = True)
        self.train_iter=mit.seekable(self.attloader)
        self.all_train_iter = mit.seekable(self.trainloader)
        self.rnd = 0
        # Load model
        self.net = torch.load('resinit').to(**setup)
        self.aggregate_weights = get_parameters(self.net)
        random.seed(self.rnd)
        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
        starts_step=0
        is_attack=True
        while is_attack==False or starts_step>0:
            starts_step-=1
            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            #for i in range(int(num_clients*subsample_rate)):
            for cid in self.cids:
                set_parameters(self.net,self.aggregate_weights)
                #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))

            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
            self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            #self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter)
        #self.state = weights_to_vector(self.aggregate_weights)
        last_layer= np.concatenate([self.aggregate_weights[-2].flatten(),self.aggregate_weights[-1]]).reshape(1,self.weights_dimension)
        state_min = np.min(last_layer)
        state_max = np.max(last_layer)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in last_layer]
        norm_state = np.array(norm_state).reshape(1,5130)
        set_parameters(self.net,self.aggregate_weights)
        self.loss, self.acc = test(self.net, self.valiloader)
        return {"pram": norm_state, "num_attacker": len(common(self.cids, att_ids))}


class FL_mnist_fltrust_g_td3_acsend_real_test_mnist(gym.Env):

    def __init__(self):
        self.rnd=0
        #self.weights_dimension=1290
        high = 1
        low = -1
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(3,),
            dtype=np.float32
        )
        self.observation_space = spaces.Dict(pram = spaces.Box(low = -np.inf, high = np.inf, shape = (1290,), dtype = np.float32),
                                             num_attacker = spaces.Discrete(11))
        self.seed()
        self.lr = 0.01
        _, self.attloader, self.valiloader, self.testloader, rootloader, att_split = load_data_fix_real(num_clients, att_ids, batch_size, 500, seed = 100, bias = 1, rootnumber = 100)
        self.root_iter = mit.seekable(rootloader)
        self.train_iter=mit.seekable(self.attloader)
        check_index = [i for i in range(200)]
        att_sub_split = Subset(att_split, check_index)
        self.att_test_loader = DataLoader(att_sub_split, batch_size = 200)
        self.att_test_iter = mit.seekable(self.att_test_loader)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.rnd+=1

        set_parameters(self.net, self.aggregate_weights)

        #action[0] = action[0]*4.9+5.0 #epsilon [0,10]
        action[0] = action[0]*0.04+0.05 #lr for local trianning [0, 0.1]
        action[1] = action[1]*10+11  #local step [1:1:50]
        #action[1] = 2
        alpha = action[-1] *0.5 + 0.5
        #action[0] = 0.1
        #action[1] = 5

        #alpha = 0.5
        new_weights=[]
        for cid in exclude(self.cids,att_ids):
            set_parameters(self.net,self.aggregate_weights)
            train(self.net, self.train_iter, epochs=1, lr=self.lr)
            #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
            new_weight=get_parameters(self.net)
            new_weights.append(new_weight)

        att_weights_lis=[]

        set_parameters(self.net, self.aggregate_weights)
        for step in range(int(action[1])):
            #att_acsend(self.net, self.aggregate_weights, self.g_weight, self.train_iter, self.lr, alpha)
            att_acsend_root(self.net, self.aggregate_weights, self.g_weight, self.att_test_iter, self.root_iter, action[0], alpha)
            att_weight = get_parameters(self.net)
            g_grad = [old - new for old, new in zip(self.aggregate_weights, self.g_weight)]
            vec_g_grad = weights_to_vector(g_grad)
            vec_att_grad = (weights_to_vector(self.aggregate_weights) - weights_to_vector(att_weight))/(np.linalg.norm(weights_to_vector(self.aggregate_weights) - weights_to_vector(att_weight)))
            vec_att_true_grad = vec_att_grad * np.linalg.norm(vec_g_grad)
            att_true_grad = vector_to_weights(vec_att_true_grad, self.aggregate_weights)
            att_true_weight = [old - grad for old, grad in zip(self.aggregate_weights, att_true_grad)]
            set_parameters(self.net, att_true_weight)

        loss, acc = test(self.net, self.valiloader)
        #print(self.rnd, loss, acc)
        # check = int(action[1])
        # while np.isnan(loss):
        #     check -= 1
        #     set_parameters(self.net, self.aggregate_weights)
        #     for step in range(check):
        #         att_acsend(self.net, self.aggregate_weights, self.g_weight, self.train_iter, self.lr, alpha)
        #     loss, acc = test(self.net, self.valiloader)
        #     print(self.rnd, loss, acc)

        # att_weight = get_parameters(self.net)
        # vec_g_grad = weights_to_vector(self.g_weight)
        # vec_att_grad = (weights_to_vector(self.aggregate_weights) - weights_to_vector(att_weight))/(np.linalg.norm(weights_to_vector(self.aggregate_weights) - weights_to_vector(att_weight)))
        # vec_att_true_grad = vec_att_grad * np.linalg.norm(vec_g_grad)
        # att_true_grad = vector_to_weights(vec_att_true_grad, self.aggregate_weights)
        # att_true_weight = [old - grad for old, grad in zip(self.aggregate_weights, att_true_grad)]
        # set_parameters(self.net, self.aggregate_weights)
        # train(self.net, self.train_iter, epochs=int(action[2]), lr=action[1], mode = False)
        # new_weight=get_parameters(self.net)
        # att_weights_lis.append(new_weight)
        # loss, acc = test(self.net, self.testloader)
        # print(self.rnd, loss, acc)


        for cid in common(self.cids, att_ids):
          new_weight = copy.deepcopy(att_true_weight)
          new_weights.append(new_weight)

        # Compute average weights
        #self.aggregate_weights =  average(new_weights)
        #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
        #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
        #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
        #print(len(new_weights))
        #print(self.cids)
        #print(len(common(self.cids, att_ids)))
        self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter, self.g_weight, lr = self.lr)
        #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)

        set_parameters(self.net,self.aggregate_weights)
        #loss, acc = test(self.net, self.testloader)
        #print(self.rnd, loss, acc)

        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))

        #is_attack=check_attack(self.cids, att_ids)
        is_attack = True
        while is_attack==False:

            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            #print('chosen clients: ', cids)
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            for i in range(int(num_clients*subsample_rate)):
                set_parameters(self.net,self.aggregate_weights)
                #train(self.net, simu_trainloader, epochs=1, lr=lr)
                #train(net, trainloaders[cid], epochs=1, lr=lr)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))

            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter, lr = self.lr)
            #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)

        set_parameters(self.net,self.aggregate_weights)
        #new_loss, _ = test(self.net, simu_testloader)
        new_loss, acc = test(self.net, self.valiloader)
        #new_loss, _ = test(self.net, testloader)
        # Caculate the reward by l(s^t(tau+1))-l(s^t(tau))
        reward = new_loss-self.loss
        #print(reward)
        #reward = self.total_cv - total_cv
        # self.total_cv = total_cv
        #reward = self.acc - acc
        #reward = -acc
        self.loss=new_loss
        self.acc = acc
        #state = np.concatenate((self.g_weight[-2], self.g_weight[-1]), axis = None)
        state = np.concatenate((self.aggregate_weights[-2], self.aggregate_weights[-1]), axis = None)
        state_min = np.min(state)
        state_max = np.max(state)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in state]
        num_attacker = len(common(self.cids, att_ids))
        set_parameters(self.net, self.aggregate_weights)
        #self.root_iter.seek(0)
        train(self.net, self.root_iter, epochs=1, lr=self.lr)
        self.g_weight=get_parameters(self.net)
        done=False

        if self.rnd>=500:
            done = True
            #print(action[:10])
            _, acc = test(self.net, self.testloader)
            print(self.rnd, new_loss, acc)
        return {"pram": norm_state, "num_attacker": num_attacker}, reward, done, {}

    def reset(self):
        #_, self.valiloader, _, rootloader = load_data_fix(batch_size=batch_size, seed = random.randint(100,10000))
        #_, self.valiloader, _, rootloader = load_data_fix(batch_size=batch_size)
        self.rnd=0
        self.net = torch.load("mnist_init").to(**setup)
        self.aggregate_weights = get_parameters(self.net)
        #self.train_iter = mit.seekable(trainloader)
        self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
        # Initial weight start from random step
        #starts_step=random.randint(0,40) #20, 100
        starts_step=0
        #print(starts_step)
        is_attack=check_attack(self.cids, att_ids)
        while is_attack==False or starts_step>0:

            starts_step-=1
            self.cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            is_attack=check_attack(self.cids, att_ids)
            new_weights=[]
            for i in range(int(num_clients*subsample_rate)):
                set_parameters(self.net,self.aggregate_weights)
                #train(self.net, simu_trainloader, epochs=1, lr=lr)
                #train(net, trainloaders[cid], epochs=1, lr=lr)
                train(self.net, self.train_iter, epochs=1, lr=self.lr)
                new_weights.append(get_parameters(self.net))
            # Compute average weights
            #self.aggregate_weights = average(new_weights)
            #self.aggregate_weights = Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)
            #self.aggregate_weights = Krum(self.aggregate_weights, new_weights, num_attacker)
            #self.aggregate_weights=Clipping(self.aggregate_weights, new_weights)
            self.aggregate_weights=FLtrust(self.aggregate_weights, new_weights, self.root_iter, lr = self.lr)
        #last_layer= np.concatenate([self.aggregate_weights[-2].flatten(),self.aggregate_weights[-1]]).reshape(1,1290)
        #self.state = pca.transform(last_layer)
        #print(self.state)
        set_parameters(self.net, self.aggregate_weights)
        self.root_iter.seek(0)
        train(self.net, self.root_iter, epochs=1, lr=self.lr)
        self.g_weight=get_parameters(self.net)

        state = np.concatenate((self.aggregate_weights[-2], self.aggregate_weights[-1]), axis = None)
        num_attacker = len(common(self.cids, att_ids))
        state_min = np.min(state)
        state_max = np.max(state)
        norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in state]
        num_attacker = len(common(self.cids, att_ids))
        set_parameters(self.net,self.aggregate_weights)
        #self.loss, _ = test(self.net, simu_testloader)
        self.loss, self.acc = test(self.net, self.valiloader)
        #self.loss, _ = test(self.net, testloader)

        return {"pram": norm_state, "num_attacker": num_attacker}
