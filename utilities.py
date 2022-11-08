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


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def default_loader(path):
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        #transforms.CenterCrop(370),
        #transforms.CenterCrop(280),
        transforms.Scale(28),
        #transforms.CenterCrop(28),
        transforms.ToTensor()
    ])
    img_pil =  Image.open(path)
    #print(img_pil)
    img_pil = img_pil.crop((143,58,512,426))
    img_tensor = preprocess(img_pil)
    return img_tensor

class Distribution_set(Dataset):
    def __init__(self, datapath='fashion_test/no_process', labelpath = 'fashion_test/data.csv', loader=default_loader):
        self.path = datapath
        files = os.listdir(self.path)
        num_png = len(files)
        self.images = []
        self.target = []
        self.loader = loader
        file = open(labelpath)
        reader = csv.reader(file)
        labels = []
        for row in reader:
            labels.append(row)
        print(len(labels))
        file.close()
        for i in range(num_png):
            if i % 1000 == 0:
                print("processing image ", i)
            fn = str(i)+'.png'
            img = self.loader(self.path+'/'+fn)
            self.target.append(int(labels[i][1]))
            self.images.append(img)

    def __getitem__(self, index):
        #fn = str(index)+'.png'
        #img = self.loader(self.path+'/'+fn)
        #target = int(self.target[index][1])
        return self.images[index],self.target[index]

    def __len__(self):
        return len(self.images)

def load_data(batch_size):
    """Load MNIST (training and test set)."""
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=apply_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=apply_transform)

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, val_indices, _, _ = train_test_split(
        range(len(train_dataset)),
        train_dataset.targets,
        stratify=train_dataset.targets,
        test_size=0.01,
    )
    #non-iid root
    # root_indices = []
    # q = 0.5
    # target_bais = 1
    # while len(root_indices)<100:
    #     index = random.randint(0, len(train_dataset)-1)
    #     if index not in root_indices:
    #         if train_dataset[index][1] == target_bais:
    #             if random.random() < q:
    #                 root_indices.append(index)
    #         else:
    #             if random.random() < (1-q)/9:
    #                 root_indices.append(index)

    _, root_indices, _, _ = train_test_split(
        range(len(train_dataset)),
        train_dataset.targets,
        stratify=train_dataset.targets,
        test_size=100
        )

    # generate subset based on indices
    train_split = Subset(train_dataset, train_indices)
    vali_split = Subset(train_dataset, val_indices)
    root_split = Subset(train_dataset, root_indices)

    counts=[0]*10
    for l in root_indices:
        counts[train_dataset[l][1]]+=1
    print(counts)
    # while np.any(np.asarray(counts)):
    #     root_indices = []
    #     q = 0.5
    #     target_bais = 1
    #     while len(root_indices)<=100:
    #         index = random.randint(0, len(train_dataset)-1)
    #         if index not in root_indices:
    #             if train_dataset[index][1] == target_bais:
    #                 if random.random() < q:
    #                     root_indices.append(index)
    #             else:
    #                 if random.random() < (1-q)/9:
    #                     root_indices.append(index)


    trainloader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    valiloader = DataLoader(vali_split, batch_size=len(vali_split)) #parallel test the whole batch
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    rootloader = DataLoader(root_split, batch_size=100)
    return trainloader, valiloader, testloader, rootloader


def seed_worker(worker_id):
    worker_seed = 1 % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def load_data_fix(batch_size, seed = 100, bias = 1, rootnumber = 100):
    """Load MNIST (training and test set)."""

    #torch.backends.cudnn.benchmark = False
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=apply_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=apply_transform)

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, val_indices, _, _ = train_test_split(
        range(len(train_dataset)),
        train_dataset.targets,
        stratify=train_dataset.targets,
        test_size=0.01,
        random_state = 0
    )

    #iid root data
    # _, root_indices, _, _ = train_test_split(
    # range(len(train_dataset)),
    # train_dataset.targets,
    # stratify=train_dataset.targets,
    # test_size=rootnumber,
    # random_state = seed
    # )

    np.random.seed(seed)
    random.seed(seed)

    #non-iid root
    root_indices = []
    q = 0.1
    target_bais = bias
    b_count = 0
    strict = True
    while len(root_indices)<rootnumber:
        index = random.randint(0, len(train_dataset)-1)
        if index not in root_indices:
            if train_dataset[index][1] == target_bais:
                if random.random() < q :
                    if strict:
                        if b_count < 100*q:
                            root_indices.append(index)
                            b_count += 1
                    else:
                        root_indices.append(index)
            else:
                if random.random() < (1-q)/9:
                    root_indices.append(index)

    #print(root_indices)


    # generate subset based on indices
    train_split = Subset(train_dataset, train_indices)
    vali_split = Subset(train_dataset, val_indices)
    root_split = Subset(train_dataset, root_indices)

    counts=[0]*10
    for l in root_indices:
        counts[train_dataset[l][1]]+=1
    print(counts)

    # np.random.seed(0)
    # random.seed(0)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)


    trainloader = DataLoader(train_split, batch_size=batch_size, shuffle=False, worker_init_fn = seed_worker, generator = g)
    valiloader = DataLoader(vali_split, batch_size=len(vali_split), worker_init_fn = seed_worker, generator = g) #parallel test the whole batch
    testloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, worker_init_fn = seed_worker, generator = g)
    rootloader = DataLoader(root_split, batch_size=100, shuffle = False, worker_init_fn = seed_worker, generator = g)
    return trainloader, valiloader, testloader, rootloader



def load_data_fix_real(num_agent, att_ids, batch_size, dataset_size, seed = 100, bias = 1, rootnumber = 100):
    """Load MNIST (training and test set)."""

    #torch.backends.cudnn.benchmark = False
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=apply_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=apply_transform)

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, _, _, _ = train_test_split(
        range(len(train_dataset)),
        train_dataset.targets,
        stratify=train_dataset.targets,
        test_size=0.05,
        random_state = 0
    )

    #iid root data
    # _, root_indices, _, _ = train_test_split(
    # range(len(train_dataset)),
    # train_dataset.targets,
    # stratify=train_dataset.targets,
    # test_size=rootnumber,
    # random_state = seed
    # )
    #
    np.random.seed(seed)
    random.seed(seed)

    #non-iid root
    root_indices = []
    q = 0.1
    target_bais = bias
    b_count = 0
    strict = True
    while len(root_indices)<rootnumber:
        index = random.randint(0, len(train_dataset)-1)
        if index not in root_indices:
            if train_dataset[index][1] == target_bais:
                if random.random() < q :
                    if strict:
                        if b_count < 100*q:
                            root_indices.append(index)
                            b_count += 1
                    else:
                        root_indices.append(index)
            else:
                if random.random() < (1-q)/9:
                    root_indices.append(index)

    #print(root_indices)

    indexes = []
    for i in range(num_agent):
        indexes.append(random.sample(train_indices, dataset_size))

    att_index = []

    for att in att_ids:
        for i in indexes[att]:
            if i not in att_index:
                att_index.append(i)

    val_index = random.sample(att_index, int(0.05*(len(att_index))))

    # generate subset based on indices
    train_split = Subset(train_dataset, train_indices)
    att_split = Subset(train_dataset, att_index)
    vali_split = Subset(train_dataset, val_index)
    root_split = Subset(train_dataset, root_indices)

    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    agent_loaders = []
    for i in range(num_agent):
        tmp = Subset(train_dataset, indexes[i])
        agent_loaders.append(DataLoader(tmp, batch_size=batch_size, shuffle=False, worker_init_fn = seed_worker, generator = g))

    counts=[0]*10
    for l in root_indices:
        counts[train_dataset[l][1]]+=1
    print(counts)

    # np.random.seed(0)
    # random.seed(0)
    trainloader = DataLoader(train_split, batch_size=batch_size, shuffle=False, worker_init_fn = seed_worker, generator = g)
    valiloader = DataLoader(vali_split, batch_size=len(vali_split), worker_init_fn = seed_worker, generator = g) #parallel test the whole batch
    testloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, worker_init_fn = seed_worker, generator = g)
    rootloader = DataLoader(root_split, batch_size=100, shuffle = False, worker_init_fn = seed_worker, generator = g)
    attloader = DataLoader(att_split, batch_size=batch_size, worker_init_fn = seed_worker, generator = g)
    return agent_loaders, attloader, valiloader, testloader, rootloader, att_split

def load_data_small(batch_size):
    """Load MNIST (training and test set)."""
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=apply_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=apply_transform)

    small_data=[]
    small_test=[]
    for i in train_dataset:
        if i[1] in [0,1,2]:
            small_data.append(i)
    for i in test_dataset:
        if i[1] in [0,1,2]:
            small_test.append(i)

    print(len(small_data))
    print(len(train_dataset))

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, val_indices = train_test_split(
        range(len(small_data)),
        test_size=0.01,
    )

    # generate subset based on indices
    train_split = Subset(small_data, train_indices)
    vali_split = Subset(small_data, val_indices)

    trainloader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    valiloader = DataLoader(vali_split, batch_size=len(vali_split)) #parallel test the whole batch
    testloader = DataLoader(small_test, batch_size=batch_size, shuffle=False)
    rootloader = DataLoader(vali_split, batch_size=100)
    return trainloader, valiloader, testloader, rootloader


def _build_mnist(data_path, augmentations=True, normalize=True):
    """Define MNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    #if mnist_mean is None:
    cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
    data_mean = (torch.mean(cc, dim=0).item(),)
    data_std = (torch.std(cc, dim=0).item(),)
    #else:
        #data_mean, data_std = mnist_mean, mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_fashion_mnist(data_path, augmentations=True, normalize=True):
    """Define FashionMNIST with everything considered."""
    # Load data
    trainset = torchvision.datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())

    #if mnist_mean is None:
    cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
    data_mean = (torch.mean(cc, dim=0).item(),)
    data_std = (torch.std(cc, dim=0).item(),)
    #else:
        #data_mean, data_std = mnist_mean, mnist_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transform])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform

    return trainset, validset

def _build_groups_by_q(trainset, q, num_class = 10):
    groups=[]
    for _ in range(num_class):
      groups.append([])
    for img,lable in trainset:
      if random.random() < (q-0.1)*num_class /(num_class-1):
        groups[lable].append((img,lable))
      else:
        groups[random.randint(0, num_class-1)].append((img,lable))
    return groups


def get_parameters(net):
    #for _, val in net.state_dict().items():
        #if np.isnan(val.cpu().numpy()).any(): print(val)
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    #print(net.state_dict().keys())
    for i in range(len(parameters)):
        if len(parameters[i].shape) == 0:
            parameters[i] = np.asarray([parameters[i]])
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    #print((state_dict))
    net.load_state_dict(state_dict, strict=True)

def train(net, train_iter, epochs, lr, mode = True):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for _ in range(epochs):
        #for images, labels in trainloader:
        try:
            images, labels = next(train_iter)
        except:
            train_iter.seek(0)
            images, labels = next(train_iter)
        #print(labels)
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if mode:
            loss = criterion(net(images), labels)
        else:
            loss = -criterion(net(images), labels)
        loss.backward()
        #clipping
        #torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

def train_real(net, trainloader, epochs, lr):
    """Train the network on the training set."""
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for _ in range(epochs):
        #for images, labels in trainloader:
        images, labels = next(iter(trainloader))
        #print(labels)
        # print('use for train')
        # plt.show(tt(images[0].cpu()))
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        #for i in range(len(images)):
              #print(dummy_labelset[i])
             # plt.subplot(1, len(images), i + 1)
             # plt.imshow(images[i].reshape(28,28).cpu().detach().numpy())
              #plt.axis('off')
              #plt.savefig("true.png")
        #labels_in = _label_to_onehot(labels, 10).to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        #dy_dx = torch.autograd.grad(loss, net.parameters(), retain_graph = True)
        #dy_dx = torch.autograd.grad(loss, net.parameters(), create_graph = True)
        loss.backward()
        optimizer.step()


def train_real_ga(net, trainloader, epochs, lr):
    """Train the network on the training set."""
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for _ in range(epochs):
        #for images, labels in trainloader:
        images, labels = next(iter(trainloader))
        #print(labels)
        # print('use for train')
        # plt.show(tt(images[0].cpu()))
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        #for i in range(len(images)):
              #print(dummy_labelset[i])
             # plt.subplot(1, len(images), i + 1)
             # plt.imshow(images[i].reshape(28,28).cpu().detach().numpy())
              #plt.axis('off')
              #plt.savefig("true.png")
        #labels_in = _label_to_onehot(labels, 10).to(DEVICE)
        optimizer.zero_grad()
        loss = -criterion(net(images), labels)
        #dy_dx = torch.autograd.grad(loss, net.parameters(), retain_graph = True)
        #dy_dx = torch.autograd.grad(loss, net.parameters(), create_graph = True)
        loss.backward()
        optimizer.step()

def test(net, valloader):
    """Validate the network on the 10% training set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in valloader:
        #data=next(iter(valloader))
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            #print(len(images))
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
    return loss, accuracy


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim = -1)


class MNISTClassifier(nn.Module):
    """
    Convolutional neural network used in the tutorial for CleverHans.
    This neural network is also used in experiments by Staib et al. (2017) and
    Sinha et al. (2018).
    """

    def __init__(self, nb_filters=64, activation='relu'):
        """
        The parameters in convolutional layers and a fully connected layer are
        initialized using the Glorot/Xavier initialization, which is the
        default initialization method in Keras.
        """

        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(1, nb_filters, kernel_size=(
            8, 8), stride=(2, 2), padding=(3, 3))
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(nb_filters, nb_filters * 2,
                               kernel_size=(6, 6), stride=(2, 2))
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(
            nb_filters * 2, nb_filters * 2, kernel_size=(5, 5), stride=(1, 1))
        nn.init.xavier_uniform_(self.conv3.weight)
        self.fc1 = nn.Linear(nb_filters * 2, 10)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.applyActivation(outputs)
        outputs = self.conv2(outputs)
        outputs = self.applyActivation(outputs)
        outputs = self.conv3(outputs)
        outputs = self.applyActivation(outputs)
        outputs = outputs.view((-1, self.num_flat_features(outputs)))
        outputs = self.fc1(outputs)
        # Note that because we use CrosEntropyLoss, which combines
        # nn.LogSoftmax and nn.NLLLoss, we do not need a softmax layer as the
        # last layer.
        return outputs

    def applyActivation(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'elu':
            return F.elu(x)
        else:
            raise ValueError("The activation function is not valid.")

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class EMNISTClassifier(nn.Module):
    """
    Convolutional neural network used in the tutorial for CleverHans.
    This neural network is also used in experiments by Staib et al. (2017) and
    Sinha et al. (2018).
    """

    def __init__(self, nb_filters=64, activation='relu'):
        """
        The parameters in convolutional layers and a fully connected layer are
        initialized using the Glorot/Xavier initialization, which is the
        default initialization method in Keras.
        """

        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(1, nb_filters, kernel_size=(
            8, 8), stride=(2, 2), padding=(3, 3))
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(nb_filters, nb_filters * 2,
                               kernel_size=(6, 6), stride=(2, 2))
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(
            nb_filters * 2, nb_filters * 2, kernel_size=(5, 5), stride=(1, 1))
        nn.init.xavier_uniform_(self.conv3.weight)
        self.fc1 = nn.Linear(nb_filters * 2, 47)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.applyActivation(outputs)
        outputs = self.conv2(outputs)
        outputs = self.applyActivation(outputs)
        outputs = self.conv3(outputs)
        outputs = self.applyActivation(outputs)
        outputs = outputs.view((-1, self.num_flat_features(outputs)))
        outputs = self.fc1(outputs)
        # Note that because we use CrosEntropyLoss, which combines
        # nn.LogSoftmax and nn.NLLLoss, we do not need a softmax layer as the
        # last layer.
        return outputs

    def applyActivation(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'elu':
            return F.elu(x)
        else:
            raise ValueError("The activation function is not valid.")

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# class CIFAR10Classifier(nn.Module):
#     """
#     Convolutional neural network used in the tutorial for CleverHans.
#     This neural network is also used in experiments by Staib et al. (2017) and
#     Sinha et al. (2018).
#     """
#
#     def __init__(self, nb_filters=64, activation='relu'):
#         """
#         The parameters in convolutional layers and a fully connected layer are
#         initialized using the Glorot/Xavier initialization, which is the
#         default initialization method in Keras.
#         """
#
#         super().__init__()
#         self.activation = activation
#         self.conv1 = nn.Conv2d(3, nb_filters, kernel_size=(
#             12, 12), stride=(2, 2), padding=(3, 3))
#         nn.init.xavier_uniform_(self.conv1.weight)
#         self.conv2 = nn.Conv2d(nb_filters, nb_filters * 2,
#                                kernel_size=(6, 6), stride=(2, 2))
#         nn.init.xavier_uniform_(self.conv2.weight)
#         self.conv3 = nn.Conv2d(
#             nb_filters * 2, nb_filters * 2, kernel_size=(5, 5), stride=(1, 1))
#         nn.init.xavier_uniform_(self.conv3.weight)
#         self.fc1 = nn.Linear(nb_filters * 2, 10)
#         nn.init.xavier_uniform_(self.fc1.weight)
#
#     def forward(self, x):
#         outputs = self.conv1(x)
#         outputs = self.applyActivation(outputs)
#         outputs = self.conv2(outputs)
#         outputs = self.applyActivation(outputs)
#         outputs = self.conv3(outputs)
#         outputs = self.applyActivation(outputs)
#         outputs = outputs.view((-1, self.num_flat_features(outputs)))
#         outputs = self.fc1(outputs)
#         # Note that because we use CrosEntropyLoss, which combines
#         # nn.LogSoftmax and nn.NLLLoss, we do not need a softmax layer as the
#         # last layer.
#         return outputs
#
#     def applyActivation(self, x):
#         if self.activation == 'relu':
#             return F.relu(x)
#         elif self.activation == 'elu':
#             return F.elu(x)
#         else:
#             raise ValueError("The activation function is not valid.")
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features

from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()
        self.conv1 = Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


import torch.nn.functional as func
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class MNISTClassifier_small(nn.Module):
    """
    Convolutional neural network used in the tutorial for CleverHans.
    This neural network is also used in experiments by Staib et al. (2017) and
    Sinha et al. (2018).
    """

    def __init__(self, nb_filters=64, activation='relu'):
        """
        The parameters in convolutional layers and a fully connected layer are
        initialized using the Glorot/Xavier initialization, which is the
        default initialization method in Keras.
        """

        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(1, nb_filters, kernel_size=(
            8, 8), stride=(2, 2), padding=(3, 3))
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(nb_filters, nb_filters * 2,
                               kernel_size=(6, 6), stride=(2, 2))
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv3 = nn.Conv2d(
            nb_filters * 2, nb_filters * 2, kernel_size=(5, 5), stride=(1, 1))
        nn.init.xavier_uniform_(self.conv3.weight)
        self.fc1 = nn.Linear(nb_filters * 2, 3)
        nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.applyActivation(outputs)
        outputs = self.conv2(outputs)
        outputs = self.applyActivation(outputs)
        outputs = self.conv3(outputs)
        outputs = self.applyActivation(outputs)
        outputs = outputs.view((-1, self.num_flat_features(outputs)))
        outputs = self.fc1(outputs)
        # Note that because we use CrosEntropyLoss, which combines
        # nn.LogSoftmax and nn.NLLLoss, we do not need a softmax layer as the
        # last layer.
        return outputs

    def applyActivation(self, x):
        if self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'elu':
            return F.elu(x)
        else:
            raise ValueError("The activation function is not valid.")

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def average_g(old_weight, new_weights):
        grads=[]
        for new_weight in new_weights:
            grad = [layer_old_weight-layer_new_weight for layer_old_weight,layer_new_weight in zip(old_weight, new_weight)]
            grads.append(weights_to_vector(grad))
        #print(len(grads[0]))
        vec_avg_grad = np.average(grads, axis = 0)
        #print(len(vec_avg_grad))
        avg_grad = vector_to_weights(vec_avg_grad, old_weight)
        #print(np.linalg.norm(vec_avg_grad))
        aggregate_weights = [layer_old_weight-layer_aggregate_grad for layer_old_weight,layer_aggregate_grad in zip(old_weight, avg_grad)]
        #print(aggregate_weights[-1][0])
        return aggregate_weights



def average(new_weights):
        fractions=[1/len(new_weights) for _ in range(len(new_weights))]
        fraction_total=np.sum(fractions)

        # Create a list of weights, each multiplied by the related fraction
        weighted_weights = [
            [layer * fraction for layer in weights] for weights, fraction in zip(new_weights, fractions)
        ]

        # Compute average weights of each layer
        aggregate_weights = [
            reduce(np.add, layer_updates) / fraction_total
            for layer_updates in zip(*weighted_weights)
        ]

        # all_weights = []
        # for new_weight in new_weights:
        #     all_weights.append(weights_to_vector(new_weight))
        # vec_aggregate_weights = np.average(all_weights, axis = 0)
        # aggregate_weights = vector_to_weights(vec_aggregate_weights, new_weights[0])
        #print(aggregate_weights[-1][0])
        return aggregate_weights

def Krum(old_weight, new_weights, num_round_attacker):
    """Compute Krum average."""

    grads=[]
    for new_weight in new_weights:
        grad = [layer_old_weight-layer_new_weight for layer_old_weight,layer_new_weight in zip(old_weight, new_weight)]
        grads.append(grad)

    scrs=[]
    for i in grads:
        scr=[]
        for j in grads:
            dif=weights_to_vector(i)-weights_to_vector(j)
            sco=np.linalg.norm(dif)
            scr.append(sco)
        top_k = sorted(scr)[1:len(grads)-2-num_round_attacker]
        scrs.append(sum(top_k))
    chosen_grads= grads[scrs.index(min(scrs))]
    krum_weights = [w1-w2 for w1,w2 in zip(old_weight, chosen_grads)]
    return krum_weights

def Median(old_weight, new_weights):
    """Compute Median average."""

    grads=[]
    for new_weight in new_weights:
        grad = [layer_old_weight-layer_new_weight for layer_old_weight,layer_new_weight in zip(old_weight, new_weight)]
        grads.append(grad)

    med_grad=[]
    for layer in range(len(grads[0])):
        lis=[]
        for weight in grads:
            lis.append(weight[layer])
        arr=np.array(lis)
        med_grad.append(np.median(arr,axis=0))
    Median_weights = [w1-w2 for w1,w2 in zip(old_weight, med_grad)]
    return Median_weights

def Clipping(old_weights, new_weights):
    max_norm=2
    grads=[]
    for new_weight in new_weights:
        norm_diff=np.linalg.norm(weights_to_vector(new_weight)-weights_to_vector(old_weights))
        clipped_grad = [(layer_old_weight-layer_new_weight)*min(1,max_norm/norm_diff) for layer_old_weight,layer_new_weight in zip(old_weights, new_weight)]
        grads.append(clipped_grad)


    fractions=[1/int(num_clients*subsample_rate) for _ in range(int(num_clients*subsample_rate))]
    fraction_total=np.sum(fractions)

    # Create a list of weights, each multiplied by the related fraction
    weighted_grads = [
        [layer * fraction for layer in grad] for grad, fraction in zip(grads, fractions)
    ]

    # Compute average weights of each layer
    aggregate_grad = [
        reduce(np.add, layer_updates) / fraction_total
        for layer_updates in zip(*weighted_grads)
    ]

    Centered_weights=[w1-w2 for w1,w2 in zip(old_weights, aggregate_grad)]


    return Centered_weights

def Clipping_Median(old_weights, new_weights):
    max_norm=2
    grads=[]
    for new_weight in new_weights:
        #print(len(new_weight))
        #print(len(old_weights))
        norm_diff=np.linalg.norm(weights_to_vector(old_weights)-weights_to_vector(new_weight))
        clipped_grad = [(layer_old_weight-layer_new_weight)*min(1,max_norm/norm_diff) for layer_old_weight,layer_new_weight in zip(old_weights, new_weight)]
        grads.append(clipped_grad)


    # fractions=[1/int(num_clients*subsample_rate) for _ in range(int(num_clients*subsample_rate))]
    # fraction_total=np.sum(fractions)
    #
    # # Create a list of weights, each multiplied by the related fraction
    # weighted_grads = [
    #     [layer * fraction for layer in grad] for grad, fraction in zip(grads, fractions)
    # ]
    #
    # # Compute average weights of each layer
    # aggregate_grad = [
    #     reduce(np.add, layer_updates) / fraction_total
    #     for layer_updates in zip(*weighted_grads)
    # ]

    med_grad=[]
    for layer in range(len(grads[0])):
        lis=[]
        for weight in grads:
            lis.append(weight[layer])
        arr=np.array(lis)
        med_grad.append(np.median(arr,axis=0))

    Centered_weights=[w1-w2 for w1,w2 in zip(old_weights, med_grad)]


    return Centered_weights


def FLtrust(old_weight, new_weights, valid_loader, g_weight = None, lr = 0.05):

    grads=[]
    for new_weight in new_weights:
        grad = [layer_old_weight-layer_new_weight for layer_old_weight,layer_new_weight in zip(old_weight, new_weight)]
        grads.append(grad)

    if g_weight == None:
        #net = torch.load("small_mnist_init").to(**setup)
        net = torch.load("emnist_init").to(**setup)
        set_parameters(net, old_weight)
        #valid_loader.seek(0)
        train(net, valid_loader, epochs=1, lr=lr)
        new_weight = get_parameters(net)
    else:
        new_weight=g_weight
    #print("2")
    #print(new_weight[-1])
    server_grad=[layer_old_weight-layer_new_weight for layer_old_weight,layer_new_weight in zip(old_weight, new_weight)]

    vec_grads=[weights_to_vector(grad) for grad in grads]
    vec_server_grad=weights_to_vector(server_grad)

    #TS = [cos_sim(vec_grad ,vec_server_grad) for vec_grad in vec_grads]
    #print(TS)
    TS = [relu(cos_sim(vec_grad ,vec_server_grad)) for vec_grad in vec_grads]

    normlized_vec_grads = [np.linalg.norm(vec_server_grad)/np.linalg.norm(vec_grad)*vec_grad for vec_grad in vec_grads]
    normlized_grads = [vector_to_weights(vec_grad, server_grad) for vec_grad in normlized_vec_grads]
    client_weights=[np.linalg.norm(vec_server_grad)/np.linalg.norm(vec_grad)*TC for vec_grad, TC in zip(vec_grads, TS)]

    #print(client_weights)

    TS_total=np.sum(TS)
    #print(TS)
    #if TS_total<0.5: TS_total=0

    # Create a list of weights, each multiplied by the related fraction
    weighted_grads = [
        [layer * TC for layer in grad] for grad, TC in zip(normlized_grads, TS)
    ]

    # Compute average weights of each layer
    FLtrust_grad = [
        reduce(np.add, layer_updates) / max(TS_total, 1e-8)
        for layer_updates in zip(*weighted_grads)
    ]

    FLtrust_weights = [w1-w2 for w1,w2 in zip(old_weight, FLtrust_grad)]



    return FLtrust_weights

def relu(x): return max(0.0, x)

def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity."""

    dot_product = np.dot(a, b) # x.y
    norm_a = np.linalg.norm(a) #|x|
    norm_b = np.linalg.norm(b) #|y|
    return dot_product / (norm_a * norm_b)

def weights_to_vector(weights):
    """Convert NumPy weights to 1-D Numpy array."""
    Lis=[np.ndarray.flatten(ndarray) for ndarray in weights]
    return np.concatenate(Lis, axis=0)

def vector_to_weights(vector,weights):
    """Convert 1-D Numpy array tp NumPy weights."""
    indies = np.cumsum([0]+[layer.size for layer in weights]) #indies for each layer of a weight
    Lis=[np.asarray(vector[indies[i]:indies[i+1]]).reshape(weights[i].shape) for i in range(len(weights))]
    return Lis

def check_attack(cids,att_ids):
    return  np.array([(id in att_ids) for id in cids]).any()



def ipm_craft_median(old_weights, new_weights):
    old_weights_vector = weights_to_vector(old_weights)
    new_weights_vector = [weights_to_vector(i) for i in new_weights]
    #print(len(new_weights))
    #print(np.max(new_weights, 0))
    min_grad_vector = [w1 - w2 for w1,w2 in zip(old_weights_vector, np.max(new_weights_vector, 0))] #grad*lr
    #min_grad_vector = [np.min(i) for i in weight_diff_vector]
    crafted_weight_diff_vector =  [-2*i for i in min_grad_vector] #ensure different sign between craft grad and min grad
    crafted_weight_diff = vector_to_weights(crafted_weight_diff_vector, old_weights)
    crafted_weight = [w1-w2 for w1,w2 in zip(old_weights, crafted_weight_diff)]
    return crafted_weight


def craft(old_weights, new_weights, action, b, type = "Clipping_Median"):
    #print(new_weights[-1][1])
    #zeta_max, zeta_min = b*0.0030664504, b*-0.0024578273
    #zeta_max=[zeta_layer*b for zeta_layer in zeta_max]
    #zeta_min=[zeta_layer*b for zeta_layer in zeta_min]
    weight_diff = [w1-w2 for w1,w2 in zip(old_weights, new_weights)] #weight_diff = grad*lr here
    crafted_weight_diff = [b*diff_layer* action for diff_layer in weight_diff]
    if type == "Clipping_Median":
        vec_weight_diff = weights_to_vector(crafted_weight_diff)
        crafted_weight_diff = vector_to_weights(vec_weight_diff, old_weights)
    else:
        vec_weight_diff = weights_to_vector(crafted_weight_diff)
        crafted_weight_diff = vector_to_weights(vec_weight_diff, old_weights)
    #crafted_weight_diff = [diff_layer* (action*(zeta_max-zeta_min)/abs(zeta_max)*0.5+(zeta_max+zeta_min)/abs(zeta_max)*0.5) for diff_layer in weight_diff]
    #crafted_weight_diff = [diff_layer* (action*(max_layer-min_layer)/np.maximum(np.absolute(max_layer), np.absolute(min_layer))*0.5
                                        #+(max_layer+min_layer)/np.maximum(np.absolute(max_layer), np.absolute(min_layer))*0.5)
                                        #for diff_layer, max_layer, min_layer in zip(weight_diff, zeta_max, zeta_min)]

    crafted_weight = [w1-w2 for w1,w2 in zip(old_weights, crafted_weight_diff)] #old_weight - lr*gradient
    return crafted_weight

def craft_att(old_weights, new_weights, action, b):
    #print(new_weights[-1][1])
    #zeta_max, zeta_min = b*0.0030664504, b*-0.0024578273
    #zeta_max=[zeta_layer*b for zeta_layer in zeta_max]
    #zeta_min=[zeta_layer*b for zeta_layer in zeta_min]
    weight_diff = [w1-w2 for w1,w2 in zip(old_weights, new_weights)] #weight_diff = grad*lr here
    crafted_weight_diff = [b*diff_layer* action for diff_layer in weight_diff]
    vec_weight_diff = weights_to_vector(crafted_weight_diff)
    #print(np.linalg.norm(vec_weight_diff))
    crafted_weight_diff = vector_to_weights(vec_weight_diff, old_weights)
    #crafted_weight_diff = [diff_layer* (action*(zeta_max-zeta_min)/abs(zeta_max)*0.5+(zeta_max+zeta_min)/abs(zeta_max)*0.5) for diff_layer in weight_diff]
    #crafted_weight_diff = [diff_layer* (action*(max_layer-min_layer)/np.maximum(np.absolute(max_layer), np.absolute(min_layer))*0.5
                                        #+(max_layer+min_layer)/np.maximum(np.absolute(max_layer), np.absolute(min_layer))*0.5)
                                        #for diff_layer, max_layer, min_layer in zip(weight_diff, zeta_max, zeta_min)]

    crafted_weight = [w1-w2 for w1,w2 in zip(old_weights, crafted_weight_diff)] #old_weight - lr*gradient
    return crafted_weight

def discrete_craft(old_weights, new_weights, action, b):

    weight_diff = [w1-w2 for w1,w2 in zip(old_weights, new_weights)] #weight_diff = grad*lr here
    indies = np.cumsum([0]+[layer.size for layer in weight_diff]) #indies for each layer of a weight
    #print(indies)

    #crafted_weight_diff = [b*diff_layer* action for diff_layer in weight_diff]
    crafted_weight_diff = [weight_diff[i] * action[indies[i]:indies[i+1]].reshape(weight_diff[i].shape) for i in range(len(weight_diff))]

    crafted_weight = [w1-w2 for w1,w2 in zip(old_weights, crafted_weight_diff)] #old_weight - lr*gradient
    return crafted_weight

def common(a,b):
    c = [value for value in a if value in b]
    return c

def exclude(a,b):
    c = [value for value in a if value not in b]
    return c

def Krum_craft(old_weights, weights_lis, att_ids, cids, net, train_iter , lr = 0.01):
    """Craft Krum weights."""

    temp_weights_lis = copy.deepcopy(weights_lis)
    num_clients=len(cids)
    num_attacker=len(att_ids)

    for aid in att_ids:
        set_parameters(net, old_weights)
        train(net, train_iter, epochs=1, lr=lr)
        new_weight=get_parameters(net)
        temp_weights_lis.append(new_weight)

    #aggregate_weight = Krum(old_weights, weights_lis, num_attacker)
    aggregate_weight = average(temp_weights_lis)
    sign = [np.sign(u-v) for u,v in zip (aggregate_weight,old_weights)]


    sum_distances=[]
    for i in range(num_clients-num_attacker):
        distance=[]
        for j in range(num_clients-num_attacker):
            dif=weights_to_vector(temp_weights_lis[i])-weights_to_vector(temp_weights_lis[j])
            distance.append(np.linalg.norm(dif))
        distance=sorted(distance)[:-2]
        sum_distances.append(sum(distance))

    distances=[]
    for i in range(num_attacker,num_clients):
        dif=weights_to_vector(temp_weights_lis[i])-weights_to_vector(old_weights)
        distances.append(np.linalg.norm(dif))


    upper_bound = min(sum_distances)/((0.6*num_clients-1)*math.sqrt(len(weights_to_vector(sign))))+ max(distances)/math.sqrt(len(weights_to_vector(sign)))
    ub = upper_bound
    lb = 0
    w_1=[u-ub*v for u,v in zip(old_weights,sign)]
    dis = ub-lb
    Lambda= upper_bound
    while dis > 1e-5:

        w_1=[u-Lambda*v for u,v in zip(old_weights,sign)]
        crafted_weights=temp_weights_lis[:num_clients-num_attacker]
        for i in range(num_attacker):
            crafted_weights.insert(-1,w_1)
        if (weights_to_vector(Krum(old_weights, temp_weights_lis, num_attacker))== weights_to_vector(w_1)).all() and Lambda==upper_bound: break
        if (weights_to_vector(Krum(old_weights, temp_weights_lis, num_attacker))== weights_to_vector(w_1)).all():
            lb=Lambda
            Lambda=(ub+lb)/2
        else:
            ub=Lambda
            Lambda=(ub+lb)/2
        dis=ub-lb

    return [w_1]*num_attacker

def Median_craft(old_weights, weights_lis, att_ids, cids, net, train_iter):
    """Craft Median weights."""

    temp_weights_lis = copy.deepcopy(weights_lis)
    for aid in att_ids:
        set_parameters(net, old_weights)
        #train(net, train_loader_lis[aid], epochs=1, lr=lr)
        train(net, train_iter, epochs=1, lr=lr)
        new_weight=get_parameters(net)
        temp_weights_lis.append(new_weight)

    aggregate_weight = Median(old_weights, temp_weights_lis)
    #aggregate_weight = average(weights_lis)
    sign = [np.sign(u-v) for u,v in zip (aggregate_weight,old_weights)]
    #print(sign)

    max_weight=weights_to_vector(temp_weights_lis[0])
    min_weight=weights_to_vector(temp_weights_lis[0])
    for i in range(1,len(cids)):
        max_weight=np.maximum(max_weight,weights_to_vector(temp_weights_lis[i]))
        min_weight=np.minimum(min_weight,weights_to_vector(temp_weights_lis[i]))

    b=2
    crafted_weights=[]


    for _ in range(len(att_ids)):
        crafted_weight=[]
        count=0
        for layer in sign:

            new_parameters=[]
            #print(layer.flatten())
            for parameter in layer.flatten():
                if parameter==-1. and max_weight[count]>0:
                    new_parameters.append(random.uniform(max_weight[count], b*max_weight[count]))
                if parameter==-1. and max_weight[count]<=0:
                    new_parameters.append(random.uniform(max_weight[count]/b, max_weight[count]))
                if parameter==1. and min_weight[count]>0:
                    new_parameters.append(random.uniform(min_weight[count]/b, min_weight[count]))
                if parameter==1. and min_weight[count]<=0:
                    new_parameters.append(random.uniform(b*min_weight[count], min_weight[count]))
                if parameter==0.: new_parameters.append(0)
                if np.isnan(parameter):
                    new_parameters.append(random.uniform(min_weight[count], max_weight[count]))
                count+=1
            #print(new_parameters)
            crafted_weight.append(np.array(new_parameters).reshape(layer.shape))
        crafted_weights.append(crafted_weight)
    return crafted_weights

def Median_craft_real(old_weights, weights_lis, att_ids, cids, net, agent_loaders, lr = 0.05):
    """Craft Median weights."""

    temp_weights_lis = copy.deepcopy(weights_lis)
    for aid in att_ids:
        set_parameters(net, old_weights)
        train_real(net, agent_loaders[aid], epochs=1, lr=lr)
        #train(net, train_iter, epochs=1, lr=lr)
        new_weight=get_parameters(net)
        temp_weights_lis.append(new_weight)

    aggregate_weight = Median(old_weights, temp_weights_lis)
    #aggregate_weight = average(weights_lis)
    sign = [np.sign(u-v) for u,v in zip (aggregate_weight,old_weights)]
    #print(sign)

    max_weight=weights_to_vector(temp_weights_lis[0])
    min_weight=weights_to_vector(temp_weights_lis[0])
    for i in range(1,len(cids)):
        max_weight=np.maximum(max_weight,weights_to_vector(temp_weights_lis[i]))
        min_weight=np.minimum(min_weight,weights_to_vector(temp_weights_lis[i]))

    b=5
    crafted_weights=[]


#    for _ in range(len(att_ids)):
    for _ in range(1):
        crafted_weight=[]
        count=0
        for layer in sign:

            new_parameters=[]
            #print(layer.flatten())
            for parameter in layer.flatten():
                if parameter==-1. and max_weight[count]>0:
                    new_parameters.append(random.uniform(max_weight[count], b*max_weight[count]))
                if parameter==-1. and max_weight[count]<=0:
                    new_parameters.append(random.uniform(max_weight[count]/b, max_weight[count]))
                if parameter==1. and min_weight[count]>0:
                    new_parameters.append(random.uniform(min_weight[count]/b, min_weight[count]))
                if parameter==1. and min_weight[count]<=0:
                    new_parameters.append(random.uniform(b*min_weight[count], min_weight[count]))
                if parameter==0.: new_parameters.append(0)
                if np.isnan(parameter):
                    new_parameters.append(random.uniform(min_weight[count], max_weight[count]))
                count+=1
            #print(new_parameters)
            crafted_weight.append(np.array(new_parameters).reshape(layer.shape))
        #crafted_weights.append(crafted_weight)
    crafted_weights = [crafted_weight for _ in range(len(att_ids))]
    return crafted_weights
def compute_updata_h(normlized_avec_grads, i,update, vec_server_grad, TS_normal, TS_normal_total, term_1, sign, u):

    #print('term_1',term_1)
    if update: normlized_avec_grads[i]=normlized_avec_grads[i]+0.005*u

    TS_attack=[relu(cos_sim(vec_grad ,vec_server_grad)) for vec_grad in normlized_avec_grads]
    TS_attack_total=np.sum(TS_attack)

    weighted_normlized_avec_grads=[ TC*vec_grad/max(TS_attack_total+TS_normal_total, 1e-8) for TC, vec_grad in zip(TS_attack, normlized_avec_grads)]

    weighted_normlized_nvec_grads=[ TC*vec_grad/max(TS_attack_total+TS_normal_total, 1e-8) for TC, vec_grad in zip(TS_normal, normlized_avec_grads)]

    term_2=np.linalg.norm(vec_server_grad)*sum([weights_to_vector(sign)@weighted_normlized_vec_grad for weighted_normlized_vec_grad in weighted_normlized_avec_grads])
    term_3=np.linalg.norm(vec_server_grad)*sum([weights_to_vector(sign)@weighted_normlized_vec_grad for weighted_normlized_vec_grad in weighted_normlized_nvec_grads])

    return term_1-term_2-term_3

def FLtrust_attack(old_weights, weights_lis, att_ids, cids, net, train_iter, iid_root_iter, lr):

    #num_attacker=len(common(cids, att_ids))
    #temp_weights_lis=list(weights_lis)
    temp_weights_lis=copy.deepcopy(weights_lis)
    for aid in common(cids, att_ids):
        set_parameters(net, old_weights)
        #train(net, train_loader_lis[aid], epochs=1, lr=lr)
        train_real(net, train_iter[aid], epochs=1, lr=lr)
        new_weight=get_parameters(net)
        temp_weights_lis.append(new_weight)
    #print(len(weights_lis))

    grads=[]
    for new_weight in temp_weights_lis:
        grad = [layer_old_weight-layer_new_weight for layer_old_weight,layer_new_weight in zip(old_weights, new_weight)]
        grads.append(grad)


    aggregate_weight = average(temp_weights_lis)
    sign = [np.sign(u-v) for u,v in zip (old_weights, aggregate_weight)]


    # caculate the server gradient
    set_parameters(net, old_weights)
    train(net, iid_root_iter, epochs=1, lr=lr)
    new_weight=get_parameters(net)
    server_grad=[layer_old_weight-layer_new_weight for layer_old_weight,layer_new_weight in zip(old_weights, new_weight)]


    vec_grads=[weights_to_vector(grad) for grad in grads]
    vec_server_grad=weights_to_vector(server_grad)

    TS = [relu(cos_sim(vec_grad ,vec_server_grad)) for vec_grad in vec_grads]
    TS_total=np.sum(TS)

    normlized_vec_grads = [np.linalg.norm(vec_server_grad)/np.linalg.norm(vec_grad)*vec_grad for vec_grad in vec_grads]
    #normlized_grads = [vector_to_weights(vec_grad, server_grad) for vec_grad in normlized_vec_grads]

    weighted_normlized_vec_grads=[ TC*vec_grad/max(TS_total, 1e-8) for TC, vec_grad in zip(TS, normlized_vec_grads)]

    term_1=np.linalg.norm(vec_server_grad)*sum([weights_to_vector(sign)@weighted_normlized_vec_grad for weighted_normlized_vec_grad in weighted_normlized_vec_grads])


    TS_normal=TS[:-len(common(cids, att_ids))]
    TS_normal_total=np.sum(TS_normal)
    normlized_nvec_grads=normlized_vec_grads[:-len(common(cids, att_ids))]


    # Initialize using Trim attack
    trim_crafted_weights = Median_craft_real(old_weights, weights_lis, att_ids, cids, net, train_iter)
    #print(len(trim_crafted_weights))
    trim_grads=[]
    for new_weight in trim_crafted_weights:
        grad = [layer_old_weight-layer_new_weight for layer_old_weight,layer_new_weight in zip(old_weights, new_weight)]
        trim_grads.append(grad)
    #print(len(trim_grads))
    vec_trim_grads = [weights_to_vector(grad) for grad in trim_grads]
    normlized_avec_grads=[vec_grad/np.linalg.norm(vec_grad) for vec_grad in vec_trim_grads]
    #print(len(normlized_avec_grads))

    #print(len(common(cids, att_ids)))
    for v in range(2):
      for i in range(len(common(cids, att_ids))):
        for q in range(2):
            u=np.random.normal(0, 0.5, np.size(vec_server_grad))
            update=True
            h1=compute_updata_h(normlized_avec_grads, i, update, vec_server_grad, TS_normal, TS_normal_total, term_1, sign, u)
            #print('h1: ',h1)
            update=False
            h2=compute_updata_h(normlized_avec_grads, i, update, vec_server_grad, TS_normal, TS_normal_total, term_1, sign, u)
            #print('h2: ',h2)
            normlized_avec_grads[i]=normlized_avec_grads[i]+0.01*(h1-h2)*u/0.005
            normlized_avec_grads[i]=normlized_avec_grads[i]/np.linalg.norm(normlized_avec_grads[i])

    normlized_a_grads = [vector_to_weights(vec_grad, server_grad) for vec_grad in normlized_avec_grads]
    #print(len(normlized_a_grads))

    crafted_weights = [[l1-l2 for l1,l2 in zip(old_weights, normlized_a_grad)] for normlized_a_grad in normlized_a_grads]

    #print(len(crafted_weights))

    return crafted_weights

def label_test(net, valloader,label):
    """Validate the network on the 10% training set."""
    #criterion = torch.nn.CrossEntropyLoss()
    correct, total = 0, 0
    with torch.no_grad():
        for data in valloader:
        #data=next(iter(valloader))
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            #loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += (labels == label).sum().item()
            correct += (predicted == label).sum().item()
        accuracy = correct / total
    return accuracy

def att_acsend(net, old_weights, g_weight, train_iter, lr, alpha):
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
    #vec_old_weights = torch.tensor(weights_to_vector(old_weights)).to(**setup)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    alpha = torch.tensor(alpha).to(**setup)
    for _ in range(1):
        #for images, labels in trainloader:
        try:
            images, labels = next(train_iter)
        except:
            train_iter.seek(0)
            images, labels = next(train_iter)
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss1 = criterion1(net(images), labels)
        params = []
        for param in net.parameters():
            params.append(param.view(-1))
        params = torch.cat(params)
        vec_old_weights = torch.tensor(weights_to_vector(old_weights)).to(**setup)
        grad = torch.sub(vec_old_weights, params)
        #grad = [old - new for old, new in zip(old_weights, get_parameters(net))]
        g_grad = [old - new for old, new in zip(old_weights, g_weight)]
        #print("g0 norm: ", np.linalg.norm(weights_to_vector(g_grad)))
        #print(grad)
        loss2 = criterion2(grad, torch.tensor(weights_to_vector(g_grad)).to(**setup))
        #print(loss2.item())
        # diff = torch.norm(grad) - torch.norm(torch.tensor(weights_to_vector(g_grad)).to(**setup))
        # if diff.item()>0:
        #     norm_loss = torch.exp(0.05 * diff).to(**setup)
        # else:
        #     norm_loss = torch.tensor([0]).to(**setup)

        #print("norm_loss", norm_loss.item())
        if loss2.item()<0.5 and loss2.item() != 0:
            penalty_loss = torch.exp(20 * (torch.tensor([0.5]).to(**setup)-loss2)).to(**setup)
            #print(penalty_loss.item())
            total_loss = -loss1+penalty_loss
        else:
            total_loss = -loss1
        #print("similarity: ", criterion2(grad, torch.tensor(weights_to_vector(g_grad)).to(**setup)))
        #loss2 = torch.inner(grad, torch.tensor(weights_to_vector(g_grad)).to(**setup))/((torch.norm(torch.tensor(weights_to_vector(g_grad)).to(**setup)))**2)
        #print(loss2.requires_grad)
        #total_loss = - alpha*loss2

        #total_loss = -loss1-penalty_loss
        #total_loss = -(torch.multiply(loss1, loss2))
        #total_loss.requires_grad = True
        #print(total_loss.requires_grad)
        print("loss1: ",loss1.item())
        # print("loss2: ",loss2.item())
        # #print("penalty_loss", penalty_loss)
        # print(total_loss.item())
        # print("---------------------")
        total_loss.backward()
        #torch.nn.utils.clip_grad_norm_(net.parameters())
        #clipping
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1000)
        optimizer.step()

def FW_att_acsend(net, old_weights, g_weight, train_iter, root_iter, round):
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
    #vec_old_weights = torch.tensor(weights_to_vector(old_weights)).to(**setup)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    g_grad = [old - new for old, new in zip(old_weights, g_weight)]
    g_norm = np.linalg.norm(weights_to_vector(g_grad))
    for i in range(round):
        previous_weights = get_parameters(net)
        if i == 0:
            vec_previous_s = weights_to_vector(old_weights)
        else:
            vec_previous_s = vec_s_true
        #print(i)
        for _ in range(1):
            try:
                images, labels = next(train_iter)
                images_root, labels_root = next(root_iter)
            except:
                train_iter.seek(0)
                root_iter.seek(0)
                images, labels = next(train_iter)
                images_root, labels_root = next(root_iter)
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images_root, labels_root = images_root.to(DEVICE), labels_root.to(DEVICE)
            optimizer.zero_grad()
            loss1 = criterion1(net(images), labels)
            loss_root = criterion1(net(images_root), labels_root)
            params = []
            for param in net.parameters():
                params.append(param.view(-1))
            params = torch.cat(params)
            vec_old_weights = torch.tensor(weights_to_vector(old_weights)).to(**setup)
            grad = torch.sub(vec_old_weights, params)
            loss2 = criterion2(grad, torch.tensor(weights_to_vector(g_grad)).to(**setup))
            # if loss2.item()<0.5 and loss2.item() != 0:
            #     penalty_loss = torch.exp(100 * (torch.tensor([0.5]).to(**setup)-loss2)).to(**setup)
            #     #print(penalty_loss.item())
            #     total_loss = -loss1+penalty_loss
            # else:
            #     total_loss = -loss1

            total_loss = loss1 - loss_root
            print('loss1:', loss1.item())
            print('root_loss:', loss_root.item())
            print('total_loss: ', total_loss.item())
            total_loss.backward()
            gradient = []
            params = list(net.parameters())
            for layer in params:
                gradient.append(layer.grad.cpu().detach().numpy())
            vec_gradient = weights_to_vector(gradient)
            norm_gradient = np.linalg.norm(vec_gradient)
            vec_s_gradient = vec_gradient/norm_gradient*g_norm
            vec_s_true = [s_g + s_previous for s_g, s_previous in zip(vec_s_gradient, vec_previous_s)]
            # print(g_norm)
            # print(np.linalg.norm(vec_s_gradient))
            lr = 2/(i+2)
            vec_previous_weight = weights_to_vector(previous_weights)
            add_on = ([(s - x) * lr for s,x in zip(vec_s_true, vec_previous_weight)])
            vec_new_weights = [x + r for x,r in zip(vec_previous_weight, add_on)]
            new_weights = vector_to_weights(vec_new_weights, old_weights)
            set_parameters(net, new_weights)

def att_acsend_root(net, old_weights, train_iter, root_iter, lr, alpha):
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
    #vec_old_weights = torch.tensor(weights_to_vector(old_weights)).to(**setup)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    alpha = torch.tensor(alpha).to(**setup)
    for _ in range(1):
        #for images, labels in trainloader:
        try:
            images, labels = next(train_iter)
            images_root, labels_root = next(root_iter)
        except:
            train_iter.seek(0)
            root_iter.seek(0)
            images, labels = next(train_iter)
            images_root, labels_root = next(root_iter)
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images_root, labels_root = images_root.to(DEVICE), labels_root.to(DEVICE)
        optimizer.zero_grad()
        loss1 = criterion1(net(images), labels)
        loss2 = criterion1(net(images_root), labels_root)
        # params = []
        # for param in net.parameters():
        #     params.append(param.view(-1))
        # params = torch.cat(params)
        # vec_old_weights = torch.tensor(weights_to_vector(old_weights)).to(**setup)
        # grad = torch.sub(vec_old_weights, params)
        # g_grad = [old - new for old, new in zip(old_weights, g_weight)]
        # loss3 = criterion2(grad, torch.tensor(weights_to_vector(g_grad)).to(**setup))
        # if loss3.item()<0.5 and loss2.item() != 0:
        #     penalty_loss = torch.exp(5 * (torch.tensor([0.5]).to(**setup)-loss3)).to(**setup)
        #     #print(penalty_loss.item())
        # else:
        #     penalty_loss = torch.Tensor([0]).to(**setup)

        total_loss = -(1-alpha)*loss1 + alpha*loss2
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1000000)
        optimizer.step()


def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy

DEFAULT_CONFIG = dict(signed=False,
                      boxed=True,
                      cost_fn='sim',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-1,
                      init='randn',
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss')

def _label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def _validate_config(config):
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None:
            raise ValueError(f'Deprecated key in config dict: {key}!')
    return config

class GradientReconstructor():
    """Instantiate a reconstruction algorithm."""


    def __init__(self, model, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1):
        """Initialize with algorithm setup."""
        self.config = _validate_config(config)
        self.model = model
        self.setup = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

        self.mean_std = mean_std
        self.num_images = num_images

        if self.config['scoring_choice'] == 'inception':
            self.inception = InceptionScore(batch_size=1, setup=self.setup)

        #self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.iDLG = True

    def reconstruct(self, input_data, labels, DummySet, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None):
        """Reconstruct image from gradient."""
        start_time = time.time()
        if eval:
            self.model.eval()


        stats = defaultdict(list)
        x = self._init_images(img_shape, DummySet)
        scores = torch.zeros(self.config['restarts'])

        if labels is None:
            if self.num_images == 1 and self.iDLG:
                # iDLG trick:
                last_weight_min = torch.argmin(torch.sum(input_data[-2], dim=-1), dim=-1)
                labels = last_weight_min.detach().reshape((1,)).requires_grad_(False)
                self.reconstruct_label = False
            else:
                # DLG label recovery
                # However this also improves conditioning for some LBFGS cases
                self.reconstruct_label = True

                def loss_fn(pred, labels):
                    labels = torch.nn.functional.softmax(labels, dim=-1)
                    return torch.mean(torch.sum(- labels * torch.nn.functional.log_softmax(pred, dim=-1), 1))
                self.loss_fn = loss_fn
        else:
            assert labels.shape[0] == self.num_images
            self.reconstruct_label = False

        try:
            all_labels = []
            for trial in range(self.config['restarts']):
                x_trial, labels = self._run_trial(x[trial], input_data, labels, dryrun=dryrun)
                # Finalize
                scores[trial] = self._score_trial(x_trial, input_data, labels)
                x[trial] = x_trial
                all_labels.append(labels)
                if tol is not None and scores[trial] <= tol:
                    break
                if dryrun:
                    break
        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass

        # Choose optimal result:
        if self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            x_optimal, stats = self._average_trials(x, labels, input_data, stats)
        else:
            print('Choosing optimal result ...')
            scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
            if torch.numel(scores) == 0:
                stats['opt'] = 1000
                print("badbadbadbad")
                return _, stats, _
            optimal_index = torch.argmin(scores)
            print(f'Optimal result score: {scores[optimal_index]:2.4f}')
            stats['opt'] = scores[optimal_index].item()
            x_optimal = x[optimal_index]
            label_optimal = all_labels[optimal_index]

        print(f'Total time: {time.time()-start_time}.')
        return x_optimal.detach(), stats, label_optimal

    def _init_images(self, img_shape, DummySet):
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'pre':
            #print(torch.stack([DummySet for _ in range(self.config['restarts'])]).shape)
            #return torch.unsqueeze(DummySet,0)
            return torch.stack([DummySet for _ in range(self.config['restarts'])])
        else:
            raise ValueError()

    def _run_trial(self, x_trial, input_data, labels, dryrun=False):
        x_trial.requires_grad = True
        if self.reconstruct_label:
            output_test = self.model(x_trial)
            #labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)
            labels = torch.randn((self.num_images,output_test.shape[1])).to(**self.setup).requires_grad_(True)
            #print(labels.shape)
            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial, labels], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial, labels], lr=self.config['lr'], momentum=0, nesterov=False)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial, labels], lr=self.config['lr'])
            else:
                raise ValueError()
        else:
            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial], lr=self.config['lr'], momentum=0, nesterov=False)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial], lr=self.config['lr'])
            else:
                raise ValueError()

        max_iterations = self.config['max_iterations']
        dm, ds = self.mean_std
        if self.config['lr_decay']:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[max_iterations // 2.667, max_iterations // 1.6,

                                                                         max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
        try:
            for iteration in range(max_iterations):
                closure = self._gradient_closure(optimizer, x_trial, input_data, labels)
                rec_loss = optimizer.step(closure)
                if self.config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.config['boxed']:
                        x_trial.data = torch.max(torch.min(x_trial, (1 - dm) / ds), -dm / ds)

                    if (iteration + 1 == max_iterations) or iteration % 500 == 0:
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')
                    elif iteration % 10 == 0 and self.config['optim'] == 'LBFGS':
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')

                    if (iteration + 1) % 500 == 0:
                        if self.config['filter'] == 'none':
                            pass
                        elif self.config['filter'] == 'median':
                            x_trial.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(x_trial)
                        else:
                            raise ValueError()

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        return x_trial.detach(), labels

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label):

        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            #print(label.shape)
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            rec_loss = reconstruction_costs([gradient], input_gradient,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * total_variation(x_trial)
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            x_trial.grad = None
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            return reconstruction_costs([gradient], input_gradient,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return total_variation(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)
        elif self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            return 0.0
        else:
            raise ValueError()

    def _average_trials(self, x, labels, input_data, stats):
        print(f'Computing a combined result via {self.config["scoring_choice"]} ...')
        if self.config['scoring_choice'] == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
        elif self.config['scoring_choice'] == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)

        self.model.zero_grad()
        if self.reconstruct_label:
            labels = self.model(x_optimal).softmax(dim=1)
        loss = self.loss_fn(self.model(x_optimal), labels)
        gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
        stats['opt'] = reconstruction_costs([gradient], input_data,
                                            cost_fn=self.config['cost_fn'],
                                            indices=self.config['indices'],
                                            weights=self.config['weights'])
        print(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats
def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError()

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    offset = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if cost_fn == 'l2':
                if input_gradient[i].shape != trial_gradient[i-offset].shape:
                    offset+=1
                    continue
                else:
                    costs += ((trial_gradient[i-offset] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                if input_gradient[i].shape != trial_gradient[i-offset].shape:
                    offset+=1
                    continue
                else:
                    costs -= (trial_gradient[i-offset] * input_gradient[i]).sum() * weights[i]
                    pnorm[0] += trial_gradient[i-offset].pow(2).sum() * weights[i]
                    pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   input_gradient[i].flatten(),
                                                                   0, 1e-10) * weights[i]
        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)

def sample_true(true_distribution, learned_distribution):
    # true_ids=[]
    # resultSet=[]
    # while len(true_ids) < learned_distribution.shape[0]:
    #     id=random.randint(0,true_distribution.shape[0]-1)
    #     if id not in true_ids:
    #         true_ids.append(id)
    #         resultSet.append(true_distribution[id].to(**setup))
    #return torch.stack(resultSet).to(**setup)
    return true_distribution[:learned_distribution.shape[0]].detach()

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


setup = dict(device=DEVICE, dtype=torch.float)

batch_size = 128
q = 0.1
num_clients= 100
subsample_rate= 0.1
num_attacker= 20
num_class = 10
fl_epoch=100
lr=0.01
num_class=10

#net = MNISTClassifier().to(**setup)
trainloader, valiloader, testloader, rootloader = load_data(batch_size=batch_size)
train_iter = mit.seekable(trainloader)
valid_iter= mit.seekable(valiloader)
root_iter = mit.seekable(rootloader)

random.seed(150)
att_ids=random.sample(range(num_clients),num_attacker)
att_ids=list(np.sort(att_ids, axis = None))
# att_ids= [0,1,2]
print('attacker ids: ', att_ids)

class agent():
    def __init__(self, type):
        self.a = type


def load_data_fix(batch_size, seed = 100, bias = 1, rootnumber = 100):
    """Load MNIST (training and test set)."""

    #torch.backends.cudnn.benchmark = False
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=apply_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=apply_transform)

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, val_indices, _, _ = train_test_split(
        range(len(train_dataset)),
        train_dataset.targets,
        stratify=train_dataset.targets,
        test_size=0.01,
        random_state = 0
    )

    #iid root data
    # _, root_indices, _, _ = train_test_split(
    # range(len(train_dataset)),
    # train_dataset.targets,
    # stratify=train_dataset.targets,
    # test_size=rootnumber,
    # random_state = seed
    # )

    np.random.seed(seed)
    random.seed(seed)

    #non-iid root
    root_indices = []
    q = 0.1
    target_bais = bias
    b_count = 0
    strict = True
    while len(root_indices)<rootnumber:
        index = random.randint(0, len(train_dataset)-1)
        if index not in root_indices:
            if train_dataset[index][1] == target_bais:
                if random.random() < q :
                    if strict:
                        if b_count < 100*q:
                            root_indices.append(index)
                            b_count += 1
                    else:
                        root_indices.append(index)
            else:
                if random.random() < (1-q)/9:
                    root_indices.append(index)

    #print(root_indices)


    # generate subset based on indices
    train_split = Subset(train_dataset, train_indices)
    vali_split = Subset(train_dataset, val_indices)
    root_split = Subset(train_dataset, root_indices)

    counts=[0]*10
    for l in root_indices:
        counts[train_dataset[l][1]]+=1
    print(counts)

    # np.random.seed(0)
    # random.seed(0)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)


    trainloader = DataLoader(train_split, batch_size=batch_size, shuffle=False, worker_init_fn = seed_worker, generator = g)
    valiloader = DataLoader(vali_split, batch_size=len(vali_split), worker_init_fn = seed_worker, generator = g) #parallel test the whole batch
    testloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, worker_init_fn = seed_worker, generator = g)
    rootloader = DataLoader(root_split, batch_size=100, shuffle = False, worker_init_fn = seed_worker, generator = g)
    return trainloader, valiloader, testloader, rootloader



def emnist_load_data_fix_real(num_agent, att_ids, batch_size, dataset_size, seed = 100, bias = 1, rootnumber = 100):
    """Load EMNIST (training and test set)."""

    #torch.backends.cudnn.benchmark = False
    train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transforms.ToTensor())
    #trainset, testset, dm, ds = _build_mnist('~/data', False, True)
    cc = torch.cat([train_dataset[i][0].reshape(-1) for i in range(len(train_dataset))], dim=0)
    dm = (torch.mean(cc, dim=0).item(),)
    ds = (torch.std(cc, dim=0).item(),)
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dm, ds)])
    train_dataset.transform = apply_transform
    test_dataset.transform = apply_transform

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, _, _, _ = train_test_split(
        range(len(train_dataset)),
        train_dataset.targets,
        stratify=train_dataset.targets,
        test_size=0.05,
        random_state = 0
    )
    #
    # #iid root data
    _, root_indices, _, _ = train_test_split(
    range(len(train_dataset)),
    train_dataset.targets,
    stratify=train_dataset.targets,
    test_size=rootnumber,
    random_state = seed
    )
    #
    # np.random.seed(seed)
    # random.seed(seed)

    #non-iid root
    # root_indices = []
    # q = 0.3
    # target_bais = bias
    # b_count = 0
    # strict = False
    # while len(root_indices)<rootnumber:
    #     index = random.randint(0, len(train_dataset)-1)
    #     if index not in root_indices:
    #         if train_dataset[index][1] == target_bais:
    #             if random.random() < q :
    #                 if strict:
    #                     if b_count < 100*q:
    #                         root_indices.append(index)
    #                         b_count += 1
    #                 else:
    #                     root_indices.append(index)
    #         else:
    #             if random.random() < (1-q)/46:
    #                 root_indices.append(index)

    print(root_indices)

    indexes = []
    for i in range(num_agent):
        indexes.append(random.sample(train_indices, dataset_size))

    att_index = []

    for att in att_ids:
        for i in indexes[att]:
            if i not in att_index:
                att_index.append(i)

    val_index = random.sample(att_index, int(0.05*(len(att_index))))

    # generate subset based on indices
    train_split = Subset(train_dataset, train_indices)
    att_split = Subset(train_dataset, att_index)
    vali_split = Subset(train_dataset, val_index)
    root_split = Subset(train_dataset, root_indices)

    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    agent_loaders = []
    for i in range(num_agent):
        tmp = Subset(train_dataset, indexes[i])
        agent_loaders.append(DataLoader(tmp, batch_size=batch_size, shuffle=True, worker_init_fn = seed_worker, generator = g))

    # counts=[0]*10
    # for l in root_indices:
    #     counts[train_dataset[l][1]]+=1
    # print(counts)

    # np.random.seed(0)
    # random.seed(0)
    trainloader = DataLoader(train_split, batch_size=batch_size, shuffle=False, worker_init_fn = seed_worker, generator = g)
    valiloader = DataLoader(vali_split, batch_size=len(vali_split), worker_init_fn = seed_worker, generator = g) #parallel test the whole batch
    testloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, worker_init_fn = seed_worker, generator = g)
    rootloader = DataLoader(root_split, batch_size=100, shuffle = False, worker_init_fn = seed_worker, generator = g)
    attloader = DataLoader(att_split, batch_size=batch_size, worker_init_fn = seed_worker, generator = g)
    return agent_loaders, attloader, valiloader, testloader, rootloader, att_split

def emnist_load_data_fix_real_guess(num_agent, att_ids, batch_size, dataset_size, seed = 100, bias = 1, rootnumber = 100):
    """Load EMNIST (training and test set)."""

    #torch.backends.cudnn.benchmark = False
    train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transforms.ToTensor())
    #trainset, testset, dm, ds = _build_mnist('~/data', False, True)
    cc = torch.cat([train_dataset[i][0].reshape(-1) for i in range(len(train_dataset))], dim=0)
    dm = (torch.mean(cc, dim=0).item(),)
    ds = (torch.std(cc, dim=0).item(),)
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dm, ds)])
    train_dataset.transform = apply_transform
    test_dataset.transform = apply_transform

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, _, _, _ = train_test_split(
        range(len(train_dataset)),
        train_dataset.targets,
        stratify=train_dataset.targets,
        test_size=0.05,
        random_state = 0
    )

    #iid root data
    _, att_root_indices, _, _ = train_test_split(
    range(len(train_dataset)),
    train_dataset.targets,
    stratify=train_dataset.targets,
    test_size=rootnumber,
    random_state = seed+1
    )
    #
    np.random.seed(seed)
    random.seed(seed)

    #non-iid root
    # root_indices = []
    # q = 0.3
    # target_bais = bias
    # b_count = 0
    # strict = False
    #
    # att_root_indices = []
    # while len(att_root_indices)<2 * rootnumber:
    #     index = random.randint(0, len(train_dataset)-1)
    #     if index not in att_root_indices:
    #         if train_dataset[index][1] == target_bais:
    #             if random.random() < q :
    #                 if strict:
    #                     if b_count < 100*q:
    #                         att_root_indices.append(index)
    #                         b_count += 1
    #                 else:
    #                     att_root_indices.append(index)
    #         else:
    #             if random.random() < (1-q)/46:
    #                 att_root_indices.append(index)

    #print(root_indices)

    # generate subset based on indices
    att_root_split = Subset(train_dataset, att_root_indices)

    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    counts=[0]*47
    for l in att_root_indices:
        counts[train_dataset[l][1]]+=1
    print(counts)

    # np.random.seed(0)
    # random.seed(0)
    att_rootloader = DataLoader(att_root_split, batch_size=200, worker_init_fn = seed_worker, generator = g)
    return att_rootloader
