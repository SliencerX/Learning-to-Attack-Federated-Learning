import importlib
import torch
torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.deterministic = True
#g = torch.Generator()
#g.manual_seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)#torch.backends.cudnn.benchmark = False

import argparse
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from algorithms.maddpg import MADDPG
from utilities import *
from exp_environments import *
import csv
import os
import random
import copy

from stable_baselines3 import TD3, DDPG, PPO, SAC
import more_itertools as mit





import matplotlib.pyplot as plt

def seed_worker(worker_id):
    worker_seed = 1 % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def _build_groups_by_q (trainloader, q):
    groups=[]
    for _ in range(47):
      groups.append([])
    for img,lable in trainset:
      if random.random() < (q-0.1)*47 /46:
        groups[lable].append((img,lable))
      else:
        groups[random.randint(0, 46)].append((img,lable))
    return groups


if __name__ == "__main__":
    setup = dict(device=DEVICE, dtype=torch.float)

    batch_size = 128
    q = 0.1
    num_clients= 100
    num_agent = num_clients
    subsample_rate= 0.1
    num_attacker= 20
    num_class = 10
    fl_epoch=2000
    lr=0.05
    num_class=10
    dataset_size = 1000
    random.seed(1001)
    norm="1"
    mode = "fltrust_lr0.05_noniid_0.1_emnist"
    att_ids=random.sample(range(num_clients),num_attacker)
    att_ids=list(np.sort(att_ids, axis = None))
    # Global initialization
    torch.cuda.init()
    device = torch.device("cuda")

    # Load maddpg result models
    #model = DDPG.load('try_mnist_ddpg_1/1690')

    #model = TD3.load('try_mnist_td3_fltrust_noniid_white_lr1e-7_loss_1/rl_model_34000_steps')
    #model = TD3.load('try_mnist_td3_fltrust_noniid_black_lr1e-7_loss/rl_model_120000_steps')
    #model = TD3.load('try_mnist_td3_fltrust_non_iid_black_lr1e-7_loss_1//rl_model_80000_steps')
    #model = TD3.load('try_mnist_td3_fltrust_non_iid_white_lr1e-7_loss_q0.4/rl_model_40000_steps')
    #model = TD3.load('try_mnist_td3_fltrust_non_iid_white_lr1e-7_loss_q0.3/rl_model_50000_steps')
    #model = TD3.load('try_mnist_td3_fltrust_non_iid_white_lr1e-7_loss_q0.2/rl_model_6000_steps')
    #model = TD3.load('try_mnist_td3_clipping_median_real_iid_white_lr1e-4_loss_norm0.1/rl_model_400000_steps')
    #model = TD3.load('try_fltrust_emnist/rl_model_10000_steps')

    custom_objects = {
      "learning_rate": 0.0,
      "lr_schedule": lambda _: 0.0,
      "clip_range": lambda _: 0.0,
    }
    model = TD3.load('rl_model_10000_steps', custom_objects = custom_objects)

    #model = PPO.load('plr-4_+discret_label0_reward_ppo_mnist_FLtrust_80000',  custom_objects = custom_objects)
    # Global initialization
    torch.cuda.init()
    trainset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transforms.ToTensor())
    testset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transforms.ToTensor())

    device = torch.device("cuda")
    groups=_build_groups_by_q(trainset, q)

    #net = MNISTClassifier().to(**setup)
    #fix_weights = get_parameters(net)
    #torch.save(net, "mnist_init")
    #net = torch.load("emnist_init").to(**setup)
    net = torch.load("emnist_init").to(**setup)
    #net = torch.load("small_mnist_init").to(**setup)
    #trainloader, valiloader, testloader, rootloader = load_data_fix(batch_size=batch_size)
    #agent_loaders, attloader, valiloader, testloader, rootloader, _ = emnist_load_data_fix_real(num_agent, att_ids, batch_size, dataset_size, seed = 100, bias = 1, rootnumber = 100)
    agent_loaders, attloader, valiloader, testloader, rootloader, _ = load_data_fix_real(num_agent, att_ids, batch_size, dataset_size, seed = 100, bias = 1, rootnumber = 100)
    #train_iter = mit.seekable(trainloader)
    #valid_iter= mit.seekable(valiloader)
    root_iter = mit.seekable(rootloader)
    #train_iter.seek(0)

    #att_ids = [0,1,2]
    retrain = False
    print('attacker ids: ', att_ids)

    # if os.path.exists(norm+mode+"ori.csv") and (not retrain):
    #     f = open(norm+mode+"ori.csv",'r')
    #     filereader = csv.reader(f)
    #     for row in filereader:
    #         ori_acc = row
    #     ori_acc = list(map(lambda x:float(x), ori_acc))
    #     #print(ori_acc)
    # else:
    #     print("-----------NA Train------------")
    #     ori_acc = []
    #     old_weights = get_parameters(net)
    #     for rnd in range(1000):
    #         print('---------------------------------------------------')
    #         print('rnd: ',rnd+1)
    #         random.seed(rnd)
    #         cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
    #         # print('chosen clients: ', cids)
    #         # print('selected attackers: ',common(cids, att_ids))
    #         weights_lis=[]
    #         #for cid in exclude(cids,att_ids):  #if there is an attack
    #         for cid in cids:  #NA env
    #             set_parameters(net, old_weights)
    #             train_real(net, agent_loaders[cid], epochs=1, lr=lr)
    #             #train(net, train_iter, epochs=1, lr=lr)
    #             new_weight=get_parameters(net)
    #             grad = [old-new for old, new in zip(old_weights, new_weight)]
    #             vec_grad = weights_to_vector(grad)
    #             #print(np.linalg.norm(vec_grad))
    #             weights_lis.append(new_weight)
    #             #max_norm=max(max_norm,np.linalg.norm(weights_to_vector(new_weight)-weights_to_vector(old_weights)))
    #         #IPM
    #         #if check_attack(cids, att_ids):
    #             #crafted_weights = [craft(old_weights, average(weights_lis), 1, 0)]*len(common(cids, att_ids))
    #             #for new_weight in crafted_weights:
    #                 #print(new_weight)
    #                 #weights_lis.append(new_weight)
    #                 #max_norm=max(max_norm,np.linalg.norm(weights_to_vector(new_weight)-weights_to_vector(old_weights)))
    #         #LMP
    #         #if check_attack(cids, att_ids):
    #             #crafted_weights = Median_craft(old_weights, weights_lis, common(cids,att_ids), cids, net, train_iter)
    #             #crafted_weights = Krum_craft(old_weights, weights_lis, common(cids,att_ids), cids, net, train_iter)
    #             #for new_weight in crafted_weights:
    #                 #print(new_weight)
    #                 #weights_lis.append(new_weight)
    #         #Random craft
    #         #print(max_norm)
    #         #aggregate_weights = average(weights_lis)
    #         #aggregate_weights = Median(old_weights, weights_lis)
    #         #aggregate_weights = Krum(old_weights, weights_lis, len(common(att_ids, cids)))
    #         #aggregate_weights=Clipping(old_weights, weights_lis)
    #         aggregate_weights=FLtrust(old_weights, weights_lis, root_iter, lr = lr)
    #         #aggregate_weights = Clipping_Median(old_weights, weights_lis)
    #
    #         old_weights=aggregate_weights
    #         set_parameters(net, old_weights)
    #         loss, acc = test(net, testloader)
    #         print('global_acc: ', acc, 'loss: ', loss)
    #         ori_acc.append(acc)
    #     #torch.save(net, 'pre_trained')
    #     f = open(norm+mode+"ori.csv", 'w')
    #     writer = csv.writer(f)
    #     writer.writerow(ori_acc)
    #     f.close()



    # net = torch.load("emnist_init").to(**setup)
    # #net = torch.load("mnist_init").to(**setup)
    # #net = torch.load("small_mnist_init").to(**setup)
    # agent_loaders, attloader, valiloader, testloader, rootloader,_ = emnist_load_data_fix_real(num_agent, att_ids, batch_size, dataset_size, seed = 100, bias = 1, rootnumber = 100)
    # #agent_loaders, attloader, valiloader, testloader, rootloader,_ = load_data_fix_real(num_agent, att_ids, batch_size, dataset_size, seed = 100, bias = 1, rootnumber = 100)
    # #train_iter = mit.seekable(trainloader)
    # #valid_iter= mit.seekable(valiloader)
    # root_iter = mit.seekable(rootloader)
    # if os.path.exists(norm+mode+"ipm.csv") and (False):
    #     f = open(norm+mode+"ipm.csv",'r')
    #     filereader = csv.reader(f)
    #     for row in filereader:
    #         ipm_acc = row
    #     ipm_acc = list(map(lambda x:float(x), ipm_acc))
    #     #print(ori_acc)
    # else:
    #     print("----------IPM Train--------------")
    #     ipm_acc = []
    #     old_weights = get_parameters(net)
    #     for rnd in range(1000):
    #         print('---------------------------------------------------')
    #         print('rnd: ',rnd+1)
    #         random.seed(rnd)
    #         cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
    #         while len(common(cids, att_ids)) == int(num_clients*subsample_rate):
    #             print("detect")
    #             cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
    #         print('chosen clients: ', cids)
    #         print('selected attackers: ',common(cids, att_ids))
    #         weights_lis=[]
    #         #
    #         for cid in exclude(cids,att_ids):  #if there is an attack
    #             set_parameters(net, old_weights)
    #             train_real(net, agent_loaders[cid], epochs=1, lr=lr)
    #             new_weight=get_parameters(net)
    #             weights_lis.append(new_weight)
    #             #max_norm=max(max_norm,np.linalg.norm(weights_to_vector(new_weight)-weights_to_vector(old_weights)))
    #         #IPM
    #         if check_attack(cids, att_ids):
    #             for i in range(len(common(cids, att_ids))):
    #                 set_parameters(net, old_weights)
    #                 #train(net, train_iter, epochs=1, lr=lr)
    #             if len(weights_lis)!=0:
    #                 # for new_weight in weights_lis:
    #                 #     print(np.linalg.norm(weights_to_vector([old_layer - new_layer for old_layer, new_layer in zip(old_weights, new_weight)])))
    #                 #craft(old_weights, average_g(old_weights, weights_lis), 1, -1)
    #                 #craft(old_weights, average(weights_lis), 1, -1)
    #                 #print("-------------------------------------------------")
    #                 crafted_weights = [craft(old_weights, average(weights_lis), 5, -1)]*len(common(cids, att_ids))
    #                 #crafted_weights = [ipm_craft_median(old_weights, weights_lis)]*len(common(cids, att_ids))
    #             else:
    #                 #crafted_weights = [ipm_craft_median(old_weights, get_parameters(net))]*len(common(cids, att_ids))
    #                 crafted_weights = [craft(old_weights, get_parameters(net), 5, -1)]*len(common(cids, att_ids))
    #             for new_weight in crafted_weights:
    #                 #print(new_weight)
    #                 weights_lis.append(new_weight)
    #         #print(len(weights_lis))
    #                 #max_norm=max(max_norm,np.linalg.norm(weights_to_vector(new_weight)-weights_to_vector(old_weights)))
    #         #LMP
    #         #if check_attack(cids, att_ids):
    #             #crafted_weights = Median_craft(old_weights, weights_lis, common(cids,att_ids), cids, net, train_iter)
    #             #crafted_weights = Krum_craft(old_weights, weights_lis, common(cids,att_ids), cids, net, train_iter)
    #             #for new_weight in crafted_weights:
    #                 #print(new_weight)
    #                 #weights_lis.append(new_weight)
    #         #Random craft
    #         #print(max_norm)
    #         #aggregate_weights = average(weights_lis)
    #         #print(crafted_weights[0][-2][0])
    #         #print(old_weights[-2][0])
    #         #aggregate_weights = Median(old_weights, weights_lis)
    #         #aggregate_weights = Krum(old_weights, weights_lis, len(common(att_ids, cids)))
    #         #aggregate_weights=Clipping(old_weights, weights_lis)
    #         aggregate_weights=FLtrust(old_weights, weights_lis, root_iter, lr = lr)
    #         #aggregate_weights=Clipping_Median(old_weights, weights_lis)
    #
    #         old_weights=aggregate_weights
    #         set_parameters(net, old_weights)
    #         loss, acc = test(net, testloader)
    #         print('global_acc: ', acc, 'loss: ', loss)
    #         ipm_acc.append(acc)
    #         root_loss, root_acc = test(net, rootloader)
    #         print("root_acc is ", root_acc)
    #     f = open(norm+mode+"ipm_new.csv", 'w')
    #     writer = csv.writer(f)
    #     writer.writerow(ipm_acc)
    #     f.close()
    #
    #
    # net = torch.load("mnist_init").to(**setup)
    # #net = torch.load("small_mnist_init").to(**setup)
    # agent_loaders, attloader, valiloader, testloader, rootloader, _ =load_data_fix_real(num_agent, att_ids, batch_size, dataset_size, seed = 100, bias = 1, rootnumber = 100)
    # #train_iter = mit.seekable(trainloader)
    # #valid_iter= mit.seekable(valiloader)
    # att_iter = mit.seekable(attloader)
    # root_iter = mit.seekable(rootloader)
    #
    # if os.path.exists(norm+mode+"lmp.csv") and (not retrain):
    #     f = open(norm+mode+"lmp.csv",'r')
    #     filereader = csv.reader(f)
    #     for row in filereader:
    #         lmp_acc = row
    #     lmp_acc = list(map(lambda x:float(x), lmp_acc))
    # else:
    #     print("---------------LMP Train---------------")
    #     old_weights=get_parameters(net)
    #     lmp_acc = []
    #     for rnd in range(1000):
    #         print('---------------------------------------------------')
    #         print('rnd: ',rnd+1)
    #         random.seed(rnd)
    #         cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
    #         while len(common(cids, att_ids)) == int(num_clients*subsample_rate):
    #             print("detect")
    #             cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
    #         print('chosen clients: ', cids)
    #         print('selected attackers: ',common(cids, att_ids))
    #         weights_lis=[]
    #         #
    #         for cid in exclude(cids,att_ids):  #if there is an attack
    #             set_parameters(net, old_weights)
    #             #train(net, train_iter, epochs=1, lr=lr)
    #             train_real(net, agent_loaders[cid], epochs = 1, lr = lr)
    #             new_weight=get_parameters(net)
    #             weights_lis.append(new_weight)
    #             #max_norm=max(max_norm,np.linalg.norm(weights_to_vector(new_weight)-weights_to_vector(old_weights)))
    #         #LMP
    #         if check_attack(cids, att_ids):
    #             if len(weights_lis)!=0:
    #                 crafted_weights = Median_craft_real(old_weights, weights_lis, common(cids,att_ids), cids, net, agent_loaders, lr = lr)
    #                 #crafted_weights = Krum_craft(old_weights, weights_lis, common(cids,att_ids), cids, net, att_iter, lr = lr)
    #             else:
    #                 crafted_weights = Median_craft_real(old_weights, [get_parameters(net)], common(cids,att_ids), cids, net, agent_loaders, lr =lr)
    #                 #crafted_weights = Krum_craft(old_weights, weights_lis, common(cids,att_ids), cids, net, att_iter, lr = lr)
    #             for new_weight in crafted_weights:
    #                 #print(new_weight)
    #                 weights_lis.append(new_weight)
    #         #Random craft
    #         #print(max_norm)
    #         #aggregate_weights = average(weights_lis)
    #         #aggregate_weights = Median(old_weights, weights_lis)
    #         #print(len(weights_lis))
    #         #aggregate_weights = Krum(old_weights, weights_lis, len(common(cids,att_ids)))
    #         #aggregate_weights=Clipping(old_weights, weights_lis)
    #         aggregate_weights=FLtrust(old_weights, weights_lis, root_iter, lr = lr)
    #         #aggregate_weights=Clipping_Median(old_weights, weights_lis)
    #
    #         old_weights=aggregate_weights
    #         set_parameters(net, old_weights)
    #         loss, acc = test(net, testloader)
    #         print('global_acc: ', acc, 'loss: ', loss)
    #         lmp_acc.append(acc)
    #     f = open(norm+mode+"lmp.csv", 'w')
    #     writer = csv.writer(f)
    #     writer.writerow(lmp_acc)
    #     f.close()


    # lr = 0.05
    # agent_loaders, attloader, valiloader, testloader, rootloader, _ = emnist_load_data_fix_real(num_agent, att_ids, batch_size, dataset_size, seed = 100, bias = 1, rootnumber = 100)
    # #train_iter = mit.seekable(trainloader)
    # #valid_iter= mit.seekable(valiloader)
    # root_iter = mit.seekable(rootloader)
    # root_iter.seek(0)
    # net = torch.load("emnist_init").to(**setup)
    # #net = torch.load("small_mnist_init").to(**setup)
    #
    #
    # if os.path.exists(norm+mode+'g0.csv') and (not retrain):
    #     f = open(norm+mode+'g0.csv','r')
    #     filereader = csv.reader(f)
    #     for row in filereader:
    #         g0_acc = row
    #     g0_acc = list(map(lambda x:float(x), g0_acc))
    # else:
    #     print("----------------G0_Method--------------")
    #     start_time = time.time()
    #     old_weights = get_parameters(net)
    #     g0_acc = []
    #     for rnd in range(200) :
    #         epoch_start_time = time.time()
    #         print('---------------------------------------------------')
    #         print('rnd: ',rnd+1)
    #         random.seed(rnd)
    #         cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
    #         while len(common(cids, att_ids)) == int(num_clients*subsample_rate):
    #             cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
    #         print('chosen clients: ', cids)
    #         print('selected attackers: ',common(cids, att_ids))
    #         selected_attacker = common(cids, att_ids)
    #         weights_lis=[]
    #         #_,_,_,att_root_loader = load_data_fix(batch_size = batch_size, seed = random.randint(100,200))
    #         #_,_,_,att_root_loader = load_data_fix(batch_size = batch_size, seed = 12)
    #         #att_root_iter = mit.seekable(att_root_loader)
    #         #black_box
    #         # _,_,_, attrootloader = load_data_fix(batch_size=batch_size, bias = 1, seed = rnd)
    #         # att_root_iter = mit.seekable(attrootloader)
    #         # set_parameters(net, old_weights)
    #         # root_iter.seek(0)
    #         # train(net, att_root_iter, epochs=1, lr=0.01)
    #         #white box
    #         set_parameters(net, old_weights)
    #         #root_iter.seek(0)
    #         train(net, root_iter, epochs=1, lr=lr)
    #         g_weight=get_parameters(net)
    #         for cid in cids:  #if there is an attack
    #             set_parameters(net, old_weights)
    #             #train(net, train_iter, epochs=1, lr=lr)
    #             train_real(net, agent_loaders[cid], epochs = 1, lr=lr)
    #             new_weight=get_parameters(net)
    #             tmp = copy.deepcopy(new_weight[-1])
    #             if cid in att_ids:
    #                 new_weight[:] = copy.deepcopy(g_weight[:])
    #             weights_lis.append(new_weight)
    #             #max_norm=max(max_norm,np.linalg.norm(weights_to_vector(new_weight)-weights_to_vector(old_weights)))
    #         #aggregate_weights = average(weights_lis)
    #         #aggregate_weights = Median(old_weights, weights_lis)
    #         #aggregate_weights = Krum(old_weights, weights_lis, num_attacker)
    #         #aggregate_weights=Clipping(old_weights, weights_lis)
    #         aggregate_weights=FLtrust(old_weights, weights_lis, root_iter, lr = lr)
    #         #aggregate_weights=Clipping_Median(old_weights, weights_lis)
    #
    #         old_weights=aggregate_weights
    #         set_parameters(net, old_weights)
    #         loss, acc = test(net, testloader)
    #         epoch_end_time = time.time()
    #         per_epoch_ptime = epoch_end_time - epoch_start_time
    #         print ('epoch %d , acc test %.2f%% , loss test %.2f ptime %.2fs .' \
    #             %(rnd, 100*acc, loss, per_epoch_ptime))
    #         g0_acc.append(acc)
    #     f = open(norm+mode+"g0.csv", 'w')
    #     writer = csv.writer(f)
    #     writer.writerow(g0_acc)
    #     f.close()

    # agent_loaders, attloader, valiloader, testloader, rootloader, _ =emnist_load_data_fix_real(num_agent, att_ids, batch_size, dataset_size, seed = 100, bias = 1, rootnumber = 100)
    # # # #train_iter = mit.seekable(trainloader)
    # # # #valid_iter= mit.seekable(valiloader)
    # # # root_iter = mit.seekable(rootloader)
    # # # root_iter.seek(0)
    # net = torch.load("emnist_init").to(**setup)
    # # # #net = torch.load("small_mnist_init").to(**setup)
    # # #
    # # #
    # if os.path.exists(norm+mode+'adapt.csv') and (False):
    #     f = open(norm+mode+'adapt.csv','r')
    #     filereader = csv.reader(f)
    #     for row in filereader:
    #         adapt_acc = row
    #     adapt_acc = list(map(lambda x:float(x), adapt_acc))
    # else:
    #     print("----------------Adaptative_Method--------------")
    #     start_time = time.time()
    #     old_weights = get_parameters(net)
    #     adapt_acc = []
    #     for rnd in range(1000) :
    #         epoch_start_time = time.time()
    #         print('---------------------------------------------------')
    #         print('rnd: ',rnd+1)
    #         random.seed(rnd)
    #         cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
    #         while len(common(cids, att_ids)) == int(num_clients*subsample_rate):
    #             cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
    #         print('chosen clients: ', cids)
    #         print('selected attackers: ',common(cids, att_ids))
    #         selected_attacker = common(cids, att_ids)
    #         weights_lis=[]
    #         #black_box
    #         # _,_,_, attrootloader = load_data_fix(batch_size=batch_size, bias = 1, seed = rnd)
    #         # att_root_iter = mit.seekable(attrootloader)
    #         # set_parameters(net, old_weights)
    #         # root_iter.seek(0)
    #         # train(net, att_root_iter, epochs=1, lr=0.01)
    #         #white box
    #         set_parameters(net, old_weights)
    #         #root_iter.seek(0)
    #         train(net, root_iter, epochs=1, lr=lr)
    #         g_weight=get_parameters(net)
    #
    #         for cid in exclude(cids,att_ids):  #if there is an attack
    #         #for cid in cids:  #NA env
    #             set_parameters(net, old_weights)
    #             #train(net, train_iters[cid%10], epochs=1, lr=lr) #no-iid
    #             train_real(net, agent_loaders[cid], epochs=1, lr=lr)
    #             new_weight=get_parameters(net)
    #             weights_lis.append(new_weight)
    #
    #         # adapt attack
    #         if check_attack(cids, att_ids):
    #             crafted_weights = FLtrust_attack(old_weights, weights_lis, common(cids, att_ids), cids, net, agent_loaders, root_iter, lr)
    #             for new_weight in crafted_weights:
    #                 #print('add_new_weight!!')
    #                 #print(len(new_weight))
    #                 weights_lis.append(new_weight)
    #         #aggregate_weights = average(weights_lis)
    #         #aggregate_weights = Median(old_weights, weights_lis)
    #         #aggregate_weights = Krum(old_weights, weights_lis, num_attacker)
    #         #aggregate_weights=Clipping(old_weights, weights_lis)
    #         print(len(weights_lis))
    #         aggregate_weights=FLtrust(old_weights, weights_lis, root_iter, lr = lr)
    #         #aggregate_weights=Clipping_Median(old_weights, weights_lis)
    #
    #         old_weights=aggregate_weights
    #         set_parameters(net, old_weights)
    #         loss, acc = test(net, testloader)
    #         epoch_end_time = time.time()
    #         per_epoch_ptime = epoch_end_time - epoch_start_time
    #         print ('epoch %d , acc test %.2f%% , loss test %.2f ptime %.2fs .' \
    #             %(rnd, 100*acc, loss, per_epoch_ptime))
    #         adapt_acc.append(acc)
    #         if rnd % 100 == 0:
    #             f = open(norm+mode+"adapt.csv", 'w')
    #             writer = csv.writer(f)
    #             writer.writerow(adapt_acc)
    #             f.close()
    #     f = open(norm+mode+"adapt.csv", 'w')
    #     writer = csv.writer(f)
    #     writer.writerow(adapt_acc)
    #     f.close()
    #
    agent_loaders, attloader, valiloader, testloader, rootloader, att_split = emnist_load_data_fix_real(num_agent, att_ids, batch_size, dataset_size, seed = 100, bias = 1, rootnumber = 100)
    #agent_loaders, attloader, valiloader, testloader, rootloader, att_split = load_data_fix_real(num_agent, att_ids, batch_size, dataset_size, seed = 100, bias = 1, rootnumber = 100)
    att_iter = mit.seekable(attloader)
    # #valid_iter= mit.seekable(valiloader)
    root_iter = mit.seekable(rootloader)
    root_iter.seek(0)

    # #net = torch.load("small_mnist_init").to(**setup)
    net = torch.load("emnist_init").to(**setup)
    lr = 0.05
    #net = torch.load("big_gradient").to(**setup)
    tmp_index = [i for i in range(5000)]
    att_sub_split = Subset(att_split, tmp_index)
    att_test_loader = DataLoader(att_sub_split, batch_size=500, shuffle = False)
    att_test_iter = mit.seekable(att_test_loader)
    if os.path.exists(norm+mode+"RL.csv") and (False):
        f = open(norm+mode+"RL.csv",'r')
        filereader = csv.reader(f)
        for row in filereader:
            td3_last_acc = row
        td3_last_acc = list(map(lambda x:float(x), td3_last_acc))
    else:
        print("----------------TD3_Last_black--------------")
        all_id = [i for i in range(len(groups[1]))]
        start_time = time.time()
        old_weights = get_parameters(net)
        td3_last_acc = []
        guess_id = np.random.choice(all_id, 100)

        for rnd in range(1000) :
            # guess_rootloader = emnist_load_data_fix_real_guess(num_agent, att_ids, batch_size, dataset_size, seed = rnd, bias = 1, rootnumber = 100)
            # guess_rootiter = mit.seekable(guess_rootloader)
            # guess_rootiter.seek(0)
            att_guess_root = Subset(groups[1], guess_id)
            att_guess_root_loader = DataLoader(att_guess_root, batch_size= 100)
            att_guess_root_iter = mit.seekable(att_guess_root_loader)


            epoch_start_time = time.time()
            print('---------------------------------------------------')
            print('rnd: ',rnd+1)
            random.seed(rnd)
            cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            while len(common(cids, att_ids)) == int(num_clients*subsample_rate):
                cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
            print('chosen clients: ', cids)
            print('selected attackers: ',common(cids, att_ids))
            selected_attacker = common(cids, att_ids)
            weights_lis=[]

            state = np.concatenate((old_weights[-2], old_weights[-1]), axis = None)
            #state = np.concatenate((old_weights[-1]), axis = None)
            state_min = np.min(state)
            state_max = np.max(state)
            norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in state]
            obs = {"pram": norm_state, "num_attacker": len(selected_attacker)}
            action, _ = model.predict(obs)

            action[0] = action[0]*0.04+0.05 #lr for local trianning [0, 0.1]
            action[1] = action[1]*1+5  #local step [1:1:50]
            #action[1] = 3
            alpha = action[-1]*0.1 +0.1
#            alpha = 0
            set_parameters(net, old_weights)
            #self.root_iter.seek(0)
            train(net, root_iter, epochs=1, lr=lr)
            g_weight=get_parameters(net)

            set_parameters(net, old_weights)
            train(net, att_guess_root_iter, epochs = 1, lr = lr)
            att_g_weight = get_parameters(net)

            #alpha = 0.5
            if rnd <= 100:
                new_weights=[]
                for cid in cids:
                    set_parameters(net,old_weights)
                    train_real(net, agent_loaders[cid], epochs=1, lr=lr)
                    #train(net, self.train_iters[cid%10], epochs=1, lr=lr)
                    new_weight=get_parameters(net)
                    new_weights.append(new_weight)

            else:
                print(action, alpha)
                new_weights=[]
                for cid in exclude(cids,att_ids):
                    set_parameters(net,old_weights)
                    train_real(net, agent_loaders[cid], epochs=1, lr=lr)
                    #train(net, self.train_iters[cid%10], epochs=1, lr=lr)
                    new_weight=get_parameters(net)
                    new_weights.append(new_weight)
                att_weights_lis=[]
                set_parameters(net, old_weights)
                check_weight = get_parameters(net)
                #round = 10
                for step in range(int(action[1])):
                    att_acsend_root(net, old_weights, att_test_iter, att_guess_root_iter, action[0], alpha)
                #FW_att_acsend(net, old_weights, g_weight, att_test_iter, root_iter, round)
                    att_weight = get_parameters(net)
                    # vec_att_weight = weights_to_vector(att_weight)
                    # print("gradient is ")
                    # print((np.linalg.norm(weights_to_vector(old_weights) - weights_to_vector(att_weight))))
                    # check_weight = copy.deepcopy(att_weight)
                    # g_grad = [old - new for old, new in zip(old_weights, g_weight)]
                    # vec_g_grad = weights_to_vector(g_grad)
                    # # print(np.linalg.norm(vec_g_grad))
                    # vec_att_grad = (weights_to_vector(old_weights) - weights_to_vector(att_weight))/(np.linalg.norm(weights_to_vector(old_weights) - weights_to_vector(att_weight)))
                    #
                    # vec_att_true_grad = vec_att_grad * np.linalg.norm(vec_g_grad)
                    # # print("cos is")
                    # # print(cos_sim(vec_att_true_grad, vec_g_grad))
                    # att_true_grad = vector_to_weights(vec_att_true_grad, old_weights)
                    # att_true_weight = [old - grad for old, grad in zip(old_weights, att_true_grad)]
                att_true_weight = craft(old_weights, att_weight, 1, 1)
                set_parameters(net, att_true_weight)

                loss, acc = test(net, valiloader)

            # if (len(exclude(cids,att_ids)) != 0):
            #     for _ in range(len(exclude(cids,att_ids))):
            #         set_parameters(net, old_weights)
            #         train(net, att_iter, epochs=int(action[1]), lr=action[0], mode = False)
            #         new_weight=get_parameters(net)
            #         #print('-------------')
            #         #print(np.linalg.norm(weights_to_vector(self.aggregate_weights) - weights_to_vector(new_weight)))
            #         att_weights_lis.append(new_weight)
            #
            # else:
            #     for _ in range(len(cids)):
            #         set_parameters(net, old_weights)
            #         train(net, att_iter, epochs=int(action[2]), lr=action[1], mode = False)
            #         new_weight=get_parameters(net)
            #         att_weights_lis.append(new_weight)

                for cid in common(cids, att_ids):
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
            print(len(new_weights))
            aggregate_weights=FLtrust(old_weights, new_weights, root_iter, g_weight, lr = lr)
            #self.aggregate_weights = Clipping_Median(self.aggregate_weights, new_weights)

            set_parameters(net,aggregate_weights)

            old_weights= copy.deepcopy(aggregate_weights)
            set_parameters(net, old_weights)
            loss, acc = test(net, testloader)
            root_loss, root_acc = test(net, rootloader)
            print("root_acc is ", root_acc)
            #reward = self.total_cv - total_cv
            #print(old_weights[-1])
            #print(label_test(net, testloader, 1))
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            print ('epoch %d , acc test %.2f%% , loss test %.2f ptime %.2fs .' \
                %(rnd, 100*acc, loss, per_epoch_ptime))
            td3_last_acc.append(acc)
        f = open(norm+mode+"RL_black_final_test.csv", 'w')
        writer = csv.writer(f)
        writer.writerow(td3_last_acc)
        f.close()



    agent_loaders, attloader, valiloader, testloader, rootloader, _ = load_data_fix_real(num_agent, [1,2], batch_size, 500, seed = 100, bias = 1, rootnumber = 100)
    #att_iter = mit.seekable(attloader)
    #valid_iter= mit.seekable(valiloader)
    #net = torch.load("small_mnist_init").to(**setup)
    net = torch.load("mnist_init").to(**setup)
    # net = torch.load("check60").to(**setup)
    # #old_weights = get_parameters(net)
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    att_trainset  = Distribution_set()
    #att_trainset.transform = apply_transform
    # transform = transforms.Normalize((0.1307,), (0.3081,))
    # check_index = [i for i in range(1000)]
    # checkset = Subset(att_trainset, check_index)
    # check_loader = DataLoader(checkset, batch_size = len(checkset))
    # acc, loss = test(net, check_loader)
    # print(acc, loss)
    # f = open('distribution_learned/data1.csv','w')
    # writer = csv.writer(f)
    # for i in range(len(att_trainset)):
    #     #print(att_trainset[i])
    #     image = transform(att_trainset[i][0].unsqueeze(0)).to(DEVICE)
    #     _, label  = torch.max(net(image), 1)
    #     writer.writerow([i, label.item()])
    #cc = torch.cat([att_trainset[i][0].reshape(-1) for i in range(len(att_trainset))], dim=0)
    #data_mean = (torch.mean(cc, dim=0).item(),)
    #data_std = (torch.std(cc, dim=0).item(),)
    #transform = transforms.Compose([
#        transforms.Normalize(data_mean, data_std)])
    #att_trainset.transform = transform
    sample_index = [i for i in range(200)]

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=apply_transform)
    val_set = Subset(train_dataset, sample_index)
    #val_set = Subset(att_trainset, sample_index)
    #valiloader = DataLoader(val_set, batch_size = len(val_set))
    #temploader = DataLoader(val_set, batch_size = len(val_set), shuffle = True)
    #train_real(net, temploader, epochs=1, lr=lr)
    #temp_weight = get_parameters(net)
    attloader = DataLoader(val_set, batch_size = batch_size, shuffle=True)
    #attloader = DataLoader(att_trainset, batch_size = batch_size, shuffle = True)
    #net = torch.load("check60").to(**setup)
    #train_real(net, attloader, epochs =1, lr = lr)
    #att_weight = get_parameters(net)
    #print("difference")
    #print(np.linalg.norm(weights_to_vector(att_weight) - weights_to_vector(temp_weight)))
    #att_grad = [old - new for old, new in zip(old_weights, att_weight)]
    #temp_grad = [old - new for old, new in zip(old_weights, temp_weight)]
    #print(np.linalg.norm(weights_to_vector(att_grad)))
    #print(np.linalg.norm(weights_to_vector(temp_grad)))
    #print(att_grad[-1])
    #print(temp_grad[-1])

    #attloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    #_, attloader,_,_,_,_ = load_data_fix_real(num_agent, [0,1], batch_size, 500, seed = 100, bias = 1, rootnumber = 100)

    # print("----------------TD3_Last_clipping_median--------------")
    #
    # start_time = time.time()
    # old_weights = get_parameters(net)
    # td3_last_acc = []
    #
    # for rnd in range(200) :
    #     epoch_start_time = time.time()
    #     print('---------------------------------------------------')
    #     print('rnd: ',rnd+1)
    #     random.seed(rnd)
    #     cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
    #     while len(common(cids, att_ids)) == int(num_clients*subsample_rate):
    #         cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
    #     print('chosen clients: ', cids)
    #     print('selected attackers: ',common(cids, att_ids))
    #     selected_attacker = common(cids, att_ids)
    #     weights_lis=[]
    #
    #     state = np.concatenate((old_weights[-2], old_weights[-1]), axis = None)
    #     #state = np.concatenate((old_weights[-1]), axis = None)
    #     state_min = np.min(state)
    #     state_max = np.max(state)
    #     norm_state = [2.0*((i - state_min)/(state_max-state_min))-1.0 for i in state]
    #     obs = {"pram": norm_state, "num_attacker": len(selected_attacker)}
    #     action, _ = model.predict(obs)
    #
    #     action[0] = action[0]*4.9+5.0 #epsilon [0,10]
    #     action[1] = action[1]*0.04+0.05 #lr for local trianning [0, 0.1]
    #     action[2] = action[2]*10+11  #local step [1:1:50]
    #     # action[0] = 2 #epsilon [0,10]
    #     # action[1] = 0.01 #lr for local trianning [0, 0.1]
    #     # action[2] = 5  #local step [1:1:50]
    #     print(action)
    #     new_weights=[]
    #     for cid in exclude(cids,att_ids):
    #         set_parameters(net,old_weights)
    #         train_real(net, agent_loaders[cid], epochs=1, lr=lr)
    #         #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
    #         new_weight=get_parameters(net)
    #         new_weights.append(new_weight)
    #
    #     att_weights_lis=[]
    #     set_parameters(net, old_weights)
    #     #train(self.net, self.train_iter, epochs=int(action[2]), lr=action[1], mode = False)
    #     train_real(net, attloader, epochs=int(action[2]), lr=action[1])
    #     new_weight=get_parameters(net)
    #     loss, acc = test(net, valiloader)
    #     #print(self.rnd, loss, acc)
    #     while np.isnan(loss):
    #         set_parameters(net, old_weights)
    #         train_real(net, attloader, epochs=int(1), lr=0.0001)
    #         new_weight=get_parameters(net)
    #         loss, acc = test(net, valiloader)
    #         print(rnd, loss, acc)
    #     att_weights_lis.append(new_weight)
    #     # if (len(exclude(cids,att_ids)) != 0):
    #     #     for _ in range(len(exclude(cids,att_ids))):
    #     #         set_parameters(net, old_weights)
    #     #         #train(self.net, self.train_iter, epochs=int(action[2]), lr=action[1], mode = False)
    #     #         train_real(net, attloader, epochs=int(action[2]), lr=action[1])
    #     #         new_weight=get_parameters(net)
    #     #         loss, acc = test(net, valiloader)
    #     #         #print(self.rnd, loss, acc)
    #     #         while np.isnan(loss):
    #     #             set_parameters(net, old_weights)
    #     #             train_real(net, attloader, epochs=int(1), lr=0.0001)
    #     #             new_weight=get_parameters(net)
    #     #             loss, acc = test(net, valiloader)
    #     #             print(rnd, loss, acc)
    #     #         att_weights_lis.append(new_weight)
    #     # else:
    #     #     for _ in range(len(cids)):
    #     #         set_parameters(net, old_weights)
    #     #         #train(self.net, self.train_iter, epochs=int(action[2]), lr=action[1], mode = False)
    #     #         train_real(net, attloader, epochs=int(action[2]), lr=action[1])
    #     #         new_weight=get_parameters(net)
    #     #         loss, acc = test(net, valiloader)
    #     #         #print(self.rnd, loss, acc)
    #     #         while np.isnan(loss):
    #     #             set_parameters(net, old_weights)
    #     #             train_real(net, attloader, epochs=int(1), lr=0.0001)
    #     #             new_weight=get_parameters(net)
    #     #             loss, acc = test(net, valiloader)
    #     #             print(rnd, loss, acc)
    #     #         att_weights_lis.append(new_weight)
    #
    #     # att_weights_lis=[]
    #     # set_parameters(self.net, self.aggregate_weights)
    #     # train(self.net, self.train_iter, epochs=int(action[2]), lr=action[1], mode = False)
    #     # new_weight=get_parameters(self.net)
    #     # att_weights_lis.append(new_weight)
    #     for cid in common(cids,att_ids):
    #         new_weight=craft_att(old_weights, average(att_weights_lis), -1, action[0])
    #         new_weights.append(new_weight)
    #     #print(len(new_weights))
    #     #aggregate_weights = average(weights_lis)
    #     #aggregate_weights = Median(old_weights, weights_lis)
    #     #aggregate_weights = Krum(old_weights, new_weights, len(common(cids, att_ids)))
    #     #aggregate_weights=Clipping(old_weights, weights_lis)
    #     #aggregate_weights=FLtrust(old_weights, weights_lis, root_iter, lr = lr)
    #     aggregate_weights=Clipping_Median(old_weights, new_weights)
    #
    #     old_weights=aggregate_weights
    #     set_parameters(net, old_weights)
    #     loss, acc = test(net, testloader)
    #     #reward = self.total_cv - total_cv
    #     #print(old_weights[-1])
    #     #print(label_test(net, testloader, 1))
    #     epoch_end_time = time.time()
    #     per_epoch_ptime = epoch_end_time - epoch_start_time
    #     print ('epoch %d , acc test %.2f%% , loss test %.2f ptime %.2fs .' \
    #         %(rnd, 100*acc, loss, per_epoch_ptime))
    #     td3_last_acc.append(acc)

    # trainloader, valiloader, testloader, rootloader = load_data_fix(batch_size=batch_size)
    # train_iter = mit.seekable(trainloader)
    # valid_iter= mit.seekable(valiloader)
    # root_iter = mit.seekable(rootloader)
    # train_iter.seek(0)
    # root_iter.seek(0)
    # #net = torch.load("mnist_init").to(**setup)
    # net = torch.load("small_mnist_init").to(**setup)
    # print("----------------Analysis_Method--------------")
    # if os.path.exists(norm+mode+"myopic.csv") and (not retrain):
    #     f = open(norm+mode+"myopic.csv",'r')
    #     filereader = csv.reader(f)
    #     for row in filereader:
    #         ana_acc = row
    #     ana_acc = list(map(lambda x:float(x), ana_acc))
    # else:
    #     start_time = time.time()
    #     old_weights = get_parameters(net)
    #     ana_acc = []
    #     for rnd in range(1000) :
    #         epoch_start_time = time.time()
    #         print('---------------------------------------------------')
    #         print('rnd: ',rnd+1)
    #         random.seed(rnd)
    #         cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
    #         while len(common(cids, att_ids)) == int(num_clients*subsample_rate):
    #             cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
    #         print('chosen clients: ', cids)
    #         print('selected attackers: ',common(cids, att_ids))
    #         selected_attacker = common(cids, att_ids)
    #         weights_lis=[]
    #         #_,_,_,att_root_loader = load_data_fix(batch_size = batch_size, seed = random.randint(100,200))
    #         #_,_,_,att_root_loader = load_data_fix(batch_size = batch_size, seed = 12)
    #         #att_root_iter = mit.seekable(att_root_loader)
    #         #black box
    #         # _,_,_, attrootloader = load_data_fix(batch_size=batch_size, bias = 1, seed = rnd)
    #         # att_root_iter = mit.seekable(attrootloader)
    #         # set_parameters(net, old_weights)
    #         # root_iter.seek(0)
    #         # train(net, att_root_iter, epochs=1, lr=0.01)
    #         # g_weight=get_parameters(net)
    #         #white Box
    #         set_parameters(net, old_weights)
    #         #root_iter.seek(0)
    #         train(net, root_iter, epochs=1, lr=lr)
    #         g_weight=get_parameters(net)
    #
    #         g_grad = old_weights[-1] - g_weight[-1]
    #         for cid in cids:  #if there is an attack
    #             set_parameters(net, old_weights)
    #             train(net, train_iter, epochs=1, lr=lr)
    #             new_weight=get_parameters(net)
    #             tmp = copy.deepcopy(new_weight[-1])
    #             if cid in att_ids:
    #                 new_grad = new_weight[-1] - old_weights[-1]
    #                 tmp_grad = []
    #                 #tmp = weights_to_vector(new_weight[-2:])
    #                 #new_weight = copy.deepcopy(g_weight)
    #                 #print(new_weight[-1])
    #                 for i in range(10):
    #                     if i == 1:
    #                         if new_grad[i]>0:
    #                             tmp_grad.append(-new_grad[i]+1e-5)
    #                             #tmp_grad.append(-10)
    #                         else:
    #                             tmp_grad.append(0)
    #                     else:
    #                         if new_grad[i]>0:
    #                             tmp_grad.append(0)
    #                         else:
    #                             tmp_grad.append(-new_grad[i]-1e-5)
    #                             #tmp_grad.append(10)
    #                 new_weight[-1] = old_weights[-1] - tmp_grad
    #
    #                 new_weight[:-1] = copy.deepcopy(g_weight[:-1])
    #             weights_lis.append(new_weight)
    #             #max_norm=max(max_norm,np.linalg.norm(weights_to_vector(new_weight)-weights_to_vector(old_weights)))
    #         #aggregate_weights = average(weights_lis)
    #         #aggregate_weights = Median(old_weights, weights_lis)
    #         #aggregate_weights = Krum(old_weights, weights_lis, num_attacker)
    #         #aggregate_weights=Clipping(old_weights, weights_lis)
    #         aggregate_weights=FLtrust(old_weights, weights_lis, root_iter)
    #         #aggregate_weights=Clipping_Median(old_weights, weights_lis)
    #
    #         old_weights=aggregate_weights
    #         set_parameters(net, old_weights)
    #         loss, acc = test(net, testloader)
    #         epoch_end_time = time.time()
    #         per_epoch_ptime = epoch_end_time - epoch_start_time
    #         print ('epoch %d , acc test %.2f%% , loss test %.2f ptime %.2fs .' \
    #             %(rnd, 100*acc, loss, per_epoch_ptime))
    #         ana_acc.append(acc)
    #     f = open(norm+mode+"myopic.csv", 'w')
    #     writer = csv.writer(f)
    #     writer.writerow(ana_acc)
    #     f.close()


    plt.plot(range(len(ori_acc)), ori_acc, label = "NA")
    plt.plot(range(len(ipm_acc)), ipm_acc, label = "IPM")
    plt.plot(range(len(lmp_acc)), lmp_acc, label = "LMP")
    #plt.plot(range(fl_epoch), maddpg_acc[:fl_epoch], label = "TD3")
    plt.plot(range(len(td3_last_acc)), td3_last_acc, label="RL")
    #plt.plot(range(len(ana_acc)), ana_acc, label = "Analysis")
    #plt.plot(range(len(g0_acc)), g0_acc, label = "G0")
    plt.plot(range(len(adapt_acc)), adapt_acc, label = "ADAPT")
    #plt.plot(range(200), ana_acc[:200], label = "Analysis")
    #plt.plot(range(200), g0_acc[:200], label = "G0")
    plt.title("Fltrust")
    plt.xlabel("Rnd")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    ##plt.savefig("50selection.png")
    # # Play
    # for episode in range(config.max_episodes):
    #     state = env.reset()
    #     done = False
    #     episode_reward = 0
    #     frames = []
    #
    #     while not done:
    #         state = state.to(device)
    #         with torch.no_grad():
    #             action = agent(state).squeeze(0)
    #             action = torch.argmax(action, dim=1).reshape(-1)#.numpy().reshape(-1)
    #         action = action.detach().cpu().numpy()#.reshape(-1)
    #         state, reward, done, _ = env.step(action)
    #
    #         frames.append(env.render(mode="rgb_array"))
    #         episode_reward += reward
    #
    #     print("Reward for Episode {}: {:.2f}".format(episode, episode_reward))
    #     utils.save_frames_as_gif(frames, config.output_dir, episode, config.dpi)
    #
    # env.close()
