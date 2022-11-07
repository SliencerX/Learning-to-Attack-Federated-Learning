from collections import OrderedDict
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.model_selection import train_test_split
import torchvision
from collections import defaultdict
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as transforms
tt = transforms.ToPILImage()
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import time
from matplotlib import pyplot
#from geomloss import SamplesLoss
import csv
import json
import os
import scipy
import math
from utilities import *
from utilities import _build_groups_by_q
from exp_environments import *
from keras.models import Model, load_model
from stable_baselines3 import DDPG, TD3, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan
from stable_baselines3.common.env_checker import check_env
import os
import gym
import fedtrain
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
from typing import Callable

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(DEVICE)

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

class Loss:
    """Abstract class, containing necessary methods.

    Abstract class to collect information about the 'higher-level' loss function, used to train an energy-based model
    containing the evaluation of the loss function, its gradients w.r.t. to first and second argument and evaluations
    of the actual metric that is targeted.

    """

    def __init__(self):
        """Init."""
        pass

    def __call__(self, reference, argmin):
        """Return l(x, y)."""
        raise NotImplementedError()
        return value, name, format

    def metric(self, reference, argmin):
        """The actually sought metric."""
        raise NotImplementedError()
        return value, name, format

class Classification(Loss):
    """A classical NLL loss for classification. Evaluation has the softmax baked in.

    The minimized criterion is cross entropy, the actual metric is total accuracy.
    """

    def __init__(self):
        """Init with torch MSE."""
        #self.loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                 #reduce=None, reduction='mean')
        self.loss_fn = cross_entropy_for_onehot

    def __call__(self, x=None, y=None):
        """Return l(x, y)."""
        name = 'CrossEntropy_for_onehot'
        format = '1.5f'
        if x is None:
            return name, format
        else:
            #labels = _label_to_onehot(y)
            #value = self.loss_fn(x, y)
            value = self.loss_fn(x,labels)
            return value, name, format

    def metric(self, x=None, y=None):
        """The actually sought metric."""
        name = 'Accuracy'
        format = '6.2%'
        if x is None:
            return name, format
        else:
            value = (x.data.argmax(dim=1) == y).sum().float() / y.shape[0]
            return value.detach(), name, format


batch_size = 128
q = 0.1 # i.i.d level, 0.1 means i.i.d, higher means non-i.i.d
num_clients = 100
subsample_rate = 0.1
num_attacker = 20
num_class = 10 #For MNIST, Fashion-MNIST and Cifar-10
fl_epoch=200
lr=0.05
num_dummy_batch=1
dummy_batch_size=32

random.seed(150)
att_ids=random.sample(range(num_clients),num_attacker)
att_ids=list(np.sort(att_ids, axis = None))
print('attacker ids: ', att_ids)

setup = dict(device=DEVICE, dtype=torch.float)

#Mnist
#Load data
apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=apply_transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=apply_transform)

# #FashionMNIST
# apply_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.2859,), (0.3530,))])
#
# trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=apply_transform)
# testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=apply_transform)

#Emnist Balanced
# trainset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transforms.ToTensor())
# testset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transforms.ToTensor())



#trainset, testset, dm, ds = _build_mnist('~/data', False, True)
cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
dm = (torch.mean(cc, dim=0).item(),)
ds = (torch.std(cc, dim=0).item(),)
apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dm, ds)])
trainset.transform = apply_transform
testset.transform = apply_transform

groups=_build_groups_by_q(trainset, q)
#groups=_build_groups_by_q(trainset, q, num_class = 47) #for emnist
trainloaders=[]
num_group_clients=int(num_clients/num_class)
#num_group_clients = 1

for gid in range(num_class):
    num_data=int(len(groups[gid])/num_group_clients)
    for cid in range(num_group_clients):
        ids = list(range(cid*num_data, (cid+1)*num_data))
        client_trainset = torch.utils.data.Subset(groups[gid], ids)
        trainloaders.append(torch.utils.data.DataLoader(client_trainset, batch_size=batch_size, shuffle=True, drop_last=True))


testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, drop_last=False)

# randomly assign attackers to each group
attackers_distribution=[]
AttackerSet=[]
for aid in att_ids:
    attackerset = trainloaders[aid].dataset
    for img,_ in attackerset:
        attackers_distribution.append(img.to(**setup).flatten())
        AttackerSet.append(img.to(**setup))

AttackerSet=torch.stack(AttackerSet)
attackers_distribution=torch.stack(attackers_distribution)

dummy_ids=[]
DummySet=[]
DummySet_array = []
DummySet_array_no = []
learned_distribution=[]
DummySet_label = []
while len(dummy_ids) < 200:
    id=random.randint(0,len(trainset)-1)
    if id not in dummy_ids:
        dummy_ids.append(id)
        DummySet.append(trainset[id][0].to(**setup))
        DummySet_array.append(trainset[id][0].reshape(28,28).cpu().detach().numpy())
        DummySet_array_no.append(trainset[id][0].reshape(28,28).cpu().detach().numpy())
        DummySet_label.append(trainset[id][1])
        learned_distribution.append(trainset[id][0].to(**setup).flatten())


DummySet=torch.stack(DummySet)
learned_distribution=torch.stack(learned_distribution)
# Load model
net = torch.load('mnist_init').to(**setup)
dm = torch.as_tensor(dm, **setup)[:, None, None]
ds = torch.as_tensor(ds, **setup)[:, None, None]

# config = dict(signed=True,
#               boxed=True,
#               cost_fn='sim',
#               indices='def',
#               weights='equal',
#               lr=0.005,
#               optim='adam',
#               restarts=1,
#               max_iterations=15000,
#               total_variation=5e-3,
#               init='randn',
#               filter='none',
#               lr_decay=True,
#               scoring_choice='loss')

#Configs of distribution learning, change to coresponding configs you want to use
#mnist Median batchsize 128
config = dict(signed=True,
              boxed=True,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=0.05,
              optim='adam',
              restarts=1,
              max_iterations=10000,
              total_variation=2e-2,
              init='zeros',
              filter='none',
              lr_decay=True,
              scoring_choice='loss')

#mnist Average batchsize 128
# config = dict(signed=True,
#               boxed=True,
#               cost_fn='sim',
#               indices='def',
#               weights='equal',
#               lr=0.05,
#               optim='adam',
#               restarts=1,
#               max_iterations=10000,
#               total_variation=6e-2,
#               init='zeros',
#               filter='none',
#               lr_decay=True,
#               scoring_choice='loss')


#mnist Krum batchsize 128
# config = dict(signed=True,
#               boxed=True,
#               cost_fn='sim',
#               indices='def',
#               weights='equal',
#               lr=0.05,
#               optim='adam',
#               restarts=1,
#               max_iterations=10000,
#               total_variation=2e-2,
#               init='zeros',
#               filter='none',
#               lr_decay=True,
#               scoring_choice='loss')

# #fashion_mnist Krum 128
# config = dict(signed=True,
#               boxed=True,
#               cost_fn='sim',
#               indices='def',
#               weights='equal',
#               lr=0.05,
#               optim='adam',
#               restarts=1,
#               max_iterations=10000,
#               total_variation=2e-2,
#               init='zeros',
#               filter='none',
#               lr_decay=True,
#               scoring_choice='loss')

#fashion_mnist clipping_median 128
# config = dict(signed=True,
#               boxed=True,
#               cost_fn='sim',
#               indices='def',
#               weights='equal',
#               lr=0.05,
#               optim='adam',
#               restarts=1,
#               max_iterations=10000,
#               total_variation=1e-2,
#               init='zeros',
#               filter='none',
#               lr_decay=True,
#               scoring_choice='loss')

#emnist clipping_median 128
# config = dict(signed=True,
#               boxed=True,
#               cost_fn='sim',
#               indices='def',
#               weights='equal',
#               lr=0.05,
#               optim='adam',
#               restarts=1,
#               max_iterations=10000,
#               total_variation=2e-3,
#               init='zeros',
#               filter='none',
#               lr_decay=True,
#               scoring_choice='loss')

#emnist krum 128
# config = dict(signed=True,
#               boxed=True,
#               cost_fn='sim',
#               indices='def',
#               weights='equal',
#               lr=0.05,
#               optim='adam',
#               restarts=1,
#               max_iterations=10000,
#               total_variation=2e-3,
#               init='zeros',
#               filter='none',
#               lr_decay=True,
#               scoring_choice='loss')


aggregate_weights = get_parameters(net)
old_weights = aggregate_weights
input_gradient = None
old_rnd=0
#set_parameters(net,initial_weights)
config_to_save = {
        "batch_size" : batch_size,
        "q" : q, # i.i.d level, 0.1 means i.i.d, higher means non-i.i.d
        "num_clients": num_clients,
        "subsample_rate": subsample_rate,
        "num_attacker": int(0.05*num_clients),
        "num_class" : num_class, #For MNIST, Fashion-MNIST and Cifar-10
        "fl_epoch":fl_epoch,
        "learning_rate":config['lr'],
        "num_dummy_batch":num_dummy_batch,
        "dummy_batch_size":dummy_batch_size,
        "signed":config['signed'],
        "boxed":config['boxed'],
        "cost_fn":config['cost_fn'],
        "indices":config['indices'],
        "weights":config['weights'],
        "optim":config['optim'],
        "restarts":config['restarts'],
        "max_iterations":config['max_iterations'],
        "total_variation":config['total_variation'],
        "init":config['init'],
        "filter":config['filter'],
        "lr_decay":config['lr_decay'],
        "scoring_choice":config['scoring_choice']}

time_stamp = time.time()
local_time = time.localtime(time_stamp)  #
str_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)

save_file_name = str_time
json_str = json.dumps(config_to_save)

for rnd in range(50):
    print('---------------------------------------------------')
    print('rnd: ',rnd)
    loss, acc = test(net, testloader)
    print('global_acc: ', acc, loss)
    cids=[]
    random.seed(rnd)
    cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
    #is_attack=check_attack(cids, att_ids)
    is_attack = True
    if rnd>0 and is_attack:
        print('attack!')
        input_gradient = [torch.from_numpy((w2-w1)/(lr*(rnd-old_rnd))).to(**setup) for w1,w2 in zip(aggregate_weights, old_weights)]
        input_gradient = [grad.detach() for grad in input_gradient]
        old_rnd=rnd
    old_weights = aggregate_weights
    if rnd>0 and input_gradient!=None:
        for i in range(num_dummy_batch):
            dummy_ids=[]
            for i in range(dummy_batch_size):
                id = random.randint(0, 500-1)
                dummy_ids.append(id)
            dummy_batch = torch.index_select(DummySet, 0, torch.tensor(dummy_ids).to(DEVICE)).to(**setup)
            rec_machine = GradientReconstructor(net, (dm, ds), config,
                                                num_images=dummy_batch_size)
            output, stats, recovered_labels = rec_machine.reconstruct(input_gradient, None, dummy_batch, img_shape=(1, 28, 28))
            output_array = []
            output_array_no = []
            for i in range(len(output)):
                output_array.append(output[i].reshape(28,28).cpu().detach().numpy())
                output_array_no.append(output[i].reshape(28,28).cpu().detach().numpy())
            # Add new recovered data to DummySet
            DummySet = torch.cat([DummySet, output], dim = 0)
            DummySet_array= np.concatenate((DummySet_array,output_array), axis=0)
            DummySet_array_no = np.concatenate((DummySet_array_no, output_array_no), axis=0)
            recovered_labels = torch.argmax(recovered_labels, dim = 1).cpu().numpy()
            DummySet_label = np.concatenate((DummySet_label, recovered_labels), axis = 0)
        set_parameters(net,aggregate_weights)

    weights_lis=[]
    ori_labels_all = []
    for cid in cids:
        set_parameters  (net,aggregate_weights)
        train_real(net, trainloaders[cid], epochs=1, lr=lr)
        weights_lis.append(get_parameters(net))
    #aggregate_weights = average(weights_lis)
    #aggregate_weights = Median(old_weights, weights_lis)
    #aggregate_weights = Krum(old_weights, weights_lis, len(common(att_ids, cids)))
    #aggregate_weights=Clipping(old_weights, weights_lis)
    #aggregate_weights=FLtrust(old_weights, weights_lis, root_iter, lr = lr)
    aggregate_weights = Clipping_Median(old_weights, weights_lis)
    #old_weights=aggregate_weights
    set_parameters(net, aggregate_weights)
    loss, acc = test(net, testloader)
    print('global_acc: ', acc, 'loss: ', loss)

f = open('experiments/mnist_clipping_median_q_0.1/data.csv','w')
#f = open('experiments/emnist_test/data.csv','w')
writer = csv.writer(f)
for i in range(len(DummySet_array)):
    writer.writerow([i, DummySet_label[i]])
    plt.close()
    plt.imshow(DummySet_array_no[i], cmap=pyplot.get_cmap('gray'))
    plt.axis('off')
    plt.savefig("experiments/mnist_clipping_median_q_0.1/no_process/"+str(i)+".png")
    plt.close()
f.close()
# #
autoencoder = load_model('autoencoder_mnist.h5')
att_trainset  = Distribution_set(datapath='experiments/mnist_clipping_median_q_0.1/no_process', labelpath = 'experiments/mnist_clipping_median_q_0.1/data.csv')
sample_index = [i for i in range(len(att_trainset))]
train_set = Subset(att_trainset, sample_index)
x_train = []
for i in train_set:
    x_train.append(i[0].numpy().reshape(28,28,1))
x_train = np.asarray(x_train)
decoded_imgs = autoencoder.predict(x_train)
for i in range(len(decoded_imgs)):
    plt.imshow(decoded_imgs[i].reshape(28,28), cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.savefig("experiments/mnist_clipping_median_q_0.1/train/"+str(i)+".png")
    plt.close()


att_trainset = Distribution_set(datapath='experiments/mnist_clipping_median_q_0.1/train', labelpath = 'experiments/mnist_clipping_median_q_0.1/data.csv')
valset = Subset(trainset, dummy_ids)
env = FL_mnist_clipping_median(att_trainset, valset)
#
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="mnist_clipping_median_q0.1/",name_prefix='rl_model')
model = TD3("MultiInputPolicy", env, buffer_size = 100000,
            policy_kwargs={"net_arch" : [256,128]}, tensorboard_log="mnist_clipping_median_q0.1/",
            verbose=1, gamma = 1, action_noise = action_noise, learning_rate=1e-7, train_freq = (5, "step"), batch_size = 256)

model.learn(total_timesteps=80000, callback = checkpoint_callback)
#


# f = open('distribution_learned_clipping_median_no/data.csv','w')
# writer = csv.writer(f)
# #     writer.writerow([wd, rnd])
# #     f.close()
# for i in range(len(DummySet_array_no[900:])):
#     writer.writerow([i, DummySet_label[900:][i]])
#     plt.imshow(DummySet_array_no[900:][i], cmap=pyplot.get_cmap('gray'))
#     plt.axis('off')
#     plt.savefig("distribution_learned_clipping_median_no/train/"+str(i)+".png")
#     #plt.savefig("distribution_learned/test/"+str(i)+".png")
#     plt.close()
# f.close()
#independtrain()
