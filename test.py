import importlib
import torch
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)
import argparse
import numpy as np
from pathlib import Path
from torch.autograd import Variable
import csv
import os
import random
import copy
from stable_baselines3 import TD3, DDPG, PPO, SAC
import more_itertools as mit
import matplotlib.pyplot as plt
from utilities import _build_groups_by_q
from utilities import *

def seed_worker(worker_id):
    worker_seed = 1 % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup = dict(device=DEVICE, dtype=torch.float)
    batch_size = 128
    q = 0.1
    num_clients= 100
    num_agent = num_clients
    subsample_rate= 0.1
    num_attacker= 20
    num_class = 10
    fl_epoch=2000
    lr=0.01
    num_class=10 #47 for emnist
    dataset_size = 500
    norm="1"
    mode = "mnist_clipping_median_q_0.1_20attacker_0.1sample_norm2"
    att_ids=random.sample(range(num_clients),num_attacker)
    att_ids=list(np.sort(att_ids, axis = None))
    # Global initialization
    torch.cuda.init()
    device = torch.device("cuda")

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
    net = torch.load("mnist_init").to(**setup)

    groups=_build_groups_by_q(trainset, q)
    #groups=_build_groups_by_q(trainset, q, num_class = 47) #for emnist

    trainloaders=[]
    num_group_clients=int(num_clients/num_class)
    for gid in range(num_class):
        num_data=int(len(groups[gid])/num_group_clients)
        for cid in range(num_group_clients):
            ids = list(range(cid*num_data, (cid+1)*num_data))
            client_trainset = torch.utils.data.Subset(groups[gid], ids)
            trainloaders.append(torch.utils.data.DataLoader(client_trainset, batch_size=batch_size, shuffle=True, drop_last=True))
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, drop_last=False)
    for seed in [1001]:
      random.seed(seed)
      att_ids=random.sample(range(num_clients),num_attacker)
      att_ids=list(np.sort(att_ids, axis = None))
      retrain = False
      print('attacker ids: ', att_ids)
      net = torch.load("mnist_init").to(**setup)
      if os.path.exists(norm+mode+str(seed)+"ori.csv") and (not retrain):
          f = open(norm+mode+str(seed)+"ori.csv",'r')
          filereader = csv.reader(f)
          for row in filereader:
              ipm_acc = row
          ipm_acc = list(map(lambda x:float(x), ipm_acc))
      else:
          print("----------Ori Train--------------")
          ori_acc = []
          old_weights = get_parameters(net)
          for rnd in range(1000):
              print('---------------------------------------------------')
              print('rnd: ',rnd+1)
              random.seed(rnd)
              cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
              while len(common(cids, att_ids)) == int(num_clients*subsample_rate):
                  print("detect")
                  cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
              print('chosen clients: ', cids)
              print('selected attackers: ',common(cids, att_ids))
              weights_lis=[]
              #
              for cid in cids:  #if there is an attack
                  set_parameters(net, old_weights)
                  train_real(net, trainloaders[cid], epochs=1, lr=lr)
                  new_weight=get_parameters(net)
                  weights_lis.append(new_weight)

              #aggregate_weights = Median(old_weights, weights_lis)
              #aggregate_weights = Krum(old_weights, weights_lis, len(common(att_ids, cids)))
              #aggregate_weights=Clipping(old_weights, weights_lis)
              #aggregate_weights=FLtrust(old_weights, weights_lis, root_iter, lr = lr)
              aggregate_weights=Clipping_Median(old_weights, weights_lis)

              old_weights=aggregate_weights
              set_parameters(net, old_weights)
              loss, acc = test(net, testloader)
              print('global_acc: ', acc, 'loss: ', loss)
              ori_acc.append(acc)
          f = open(norm+mode+str(seed)+"ori.csv", 'w')
          writer = csv.writer(f)
          writer.writerow(ori_acc)
          f.close()

      net = torch.load("mnist_init").to(**setup)
      if os.path.exists(norm+mode+str(seed)+"ipm.csv") and (not retrain):
          f = open(norm+mode+str(seed)+"ipm.csv",'r')
          filereader = csv.reader(f)
          for row in filereader:
              ipm_acc = row
          ipm_acc = list(map(lambda x:float(x), ipm_acc))
      else:
          print("----------IPM Train--------------")
          ipm_acc = []
          old_weights = get_parameters(net)
          for rnd in range(1000):
              print('---------------------------------------------------')
              print('rnd: ',rnd+1)
              random.seed(rnd)
              cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
              while len(common(cids, att_ids)) == int(num_clients*subsample_rate):
                  print("detect")
                  cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
              print('chosen clients: ', cids)
              print('selected attackers: ',common(cids, att_ids))
              weights_lis=[]
              #
              for cid in exclude(cids,att_ids):  #if there is an attack
                  set_parameters(net, old_weights)
                  train_real(net, trainloaders[cid], epochs=1, lr=lr)
                  new_weight=get_parameters(net)
                  weights_lis.append(new_weight)
                  #max_norm=max(max_norm,np.linalg.norm(weights_to_vector(new_weight)-weights_to_vector(old_weights)))
              #IPM
              if check_attack(cids, att_ids):
                  for i in range(len(common(cids, att_ids))):
                      set_parameters(net, old_weights)
                      #train(net, train_iter, epochs=1, lr=lr)
                  if len(weights_lis)!=0:
                      crafted_weights = [craft(old_weights, average(weights_lis), 5, -1)]*len(common(cids, att_ids))
                      #crafted_weights = [ipm_craft_median(old_weights, weights_lis)]*len(common(cids, att_ids))
                  else:
                      #crafted_weights = [ipm_craft_median(old_weights, get_parameters(net))]*len(common(cids, att_ids))
                      crafted_weights = [craft(old_weights, get_parameters(net), 5, -1)]*len(common(cids, att_ids))
                  for new_weight in crafted_weights:
                      #print(new_weight)
                      weights_lis.append(new_weight)
              #aggregate_weights = Median(old_weights, weights_lis)
              #aggregate_weights = Krum(old_weights, weights_lis, len(common(att_ids, cids)))
              #aggregate_weights=Clipping(old_weights, weights_lis)
              #aggregate_weights=FLtrust(old_weights, weights_lis, root_iter, lr = lr)
              aggregate_weights=Clipping_Median(old_weights, weights_lis)

              old_weights=aggregate_weights
              set_parameters(net, old_weights)
              loss, acc = test(net, testloader)
              print('global_acc: ', acc, 'loss: ', loss)
              ipm_acc.append(acc)
          f = open(norm+mode+str(seed)+"ipm.csv", 'w')
          writer = csv.writer(f)
          writer.writerow(ipm_acc)
          f.close()

      net = torch.load("mnist_init").to(**setup)
      if os.path.exists(norm+mode+str(seed)+"lmp.csv") and (not retrain):
          f = open(norm+mode+str(seed)+"lmp.csv",'r')
          filereader = csv.reader(f)
          for row in filereader:
              lmp_acc = row
          lmp_acc = list(map(lambda x:float(x), lmp_acc))
          #continue
      else:
          print("---------------LMP Train---------------")
          old_weights=get_parameters(net)
          lmp_acc = []
          for rnd in range(1000):
              epoch_start_time = time.time()
              print('---------------------------------------------------')
              print('rnd: ',rnd+1)
              random.seed(rnd)
              cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
              while len(common(cids, att_ids)) == int(num_clients*subsample_rate):
                  print("detect")
                  cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
              print('chosen clients: ', cids)
              print('selected attackers: ',common(cids, att_ids))
              weights_lis=[]
              #
              for cid in exclude(cids,att_ids):  #if there is an attack
                  set_parameters(net, old_weights)
                  train_real(net, trainloaders[cid], epochs = 1, lr = lr)
                  new_weight=get_parameters(net)
                  weights_lis.append(new_weight)
              #LMP
              if check_attack(cids, att_ids):
                  if len(weights_lis)!=0:
                      crafted_weights = Median_craft_real(old_weights, weights_lis, common(cids,att_ids), cids, net, trainloaders, lr = lr)
                      #crafted_weights = Krum_craft(old_weights, weights_lis, common(cids,att_ids), cids, net, trainloaders, lr = lr)
                  else:
                      crafted_weights = Median_craft_real(old_weights, [get_parameters(net)], common(cids,att_ids), cids, net, trainloaders, lr =lr)
                      #crafted_weights = Krum_craft(old_weights, weights_lis, common(cids,att_ids), cids, net, trainloaders, lr = lr)
                  for new_weight in crafted_weights:
                      #print(new_weight)
                      weights_lis.append(new_weight)

              #aggregate_weights = Krum(old_weights, weights_lis, len(common(cids,att_ids)))
              #aggregate_weights=Clipping(old_weights, weights_lis)
              #aggregate_weights=FLtrust(old_weights, weights_lis, root_iter, lr = lr)
              aggregate_weights=Clipping_Median(old_weights, weights_lis)
              epoch_end_time = time.time()
              per_epoch_ptime = epoch_end_time - epoch_start_time
              old_weights=aggregate_weights
              set_parameters(net, old_weights)
              loss, acc = test(net, testloader)
              print('global_acc: ', acc, 'loss: ', loss)
              lmp_acc.append(acc)
              print('time is ', str(per_epoch_ptime))
          f = open(norm+mode+str(seed)+"lmp.csv", 'w')
          writer = csv.writer(f)
          writer.writerow(lmp_acc)
          f.close()

      net = torch.load("mnist_init").to(**setup)
      if os.path.exists(norm+mode+str(seed)+"EB.csv") and (not retrain):
          f = open(norm+mode+str(seed)+"EB.csv",'r')
          filereader = csv.reader(f)
          for row in filereader:
              ipm_acc = row
          EB_acc = list(map(lambda x:float(x), ipm_acc))
          #print(ori_acc)
      else:
          print("----------EB Train--------------")
          EB_acc = []
          dummy_id = [i for i in range(1000)]
          validset = Subset(trainset, dummy_id)
          valiloader = DataLoader(validset, batch_size=200, shuffle=True)
          old_weights = get_parameters(net)
          for rnd in range(1000):
              print('---------------------------------------------------')
              print('rnd: ',rnd+1)
              random.seed(rnd)
              cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
              while len(common(cids, att_ids)) == int(num_clients*subsample_rate):
                  print("detect")
                  cids=random.sample(range(num_clients),int(num_clients*subsample_rate))
              print('chosen clients: ', cids)
              print('selected attackers: ',common(cids, att_ids))
              weights_lis=[]
              #
              for cid in exclude(cids,att_ids):  #if there is an attack
                  set_parameters(net, old_weights)
                  train_real(net, trainloaders[cid], epochs=1, lr=lr)
                  new_weight=get_parameters(net)
                  weights_lis.append(new_weight)
                  #max_norm=max(max_norm,np.linalg.norm(weights_to_vector(new_weight)-weights_to_vector(old_weights)))

              if check_attack(cids, att_ids):
                  for cid in common(cids, att_ids):
                    set_parameters(net, old_weights)
                    train_real_ga(net, trainloaders[cid], epochs = 5, lr = lr)
                    loss, acc = test(net, valiloader)
                    #print(self.rnd, loss, acc)
                    check = 5
                    while np.isnan(loss):
                        check = max(check - 1, 0)
                        set_parameters(net, old_weights)
                        train_real(net, valiloader, epochs=check, lr=lr)
                        new_weight=get_parameters(net)
                        loss, acc = test(net, valiloader)
                        print(rnd, loss, acc, check)
                        if check == 0:
                          new_weight = copy.deepcopy(old_weights)
                          break
                    weights_lis.append(craft(old_weights, new_weight, 1, len(cids)/len(common(cids, att_ids))))
              #aggregate_weights = Median(old_weights, weights_lis)
              #aggregate_weights = Krum(old_weights, weights_lis, len(common(att_ids, cids)))
              #aggregate_weights=Clipping(old_weights, weights_lis)
              #aggregate_weights=FLtrust(old_weights, weights_lis, root_iter, lr = lr)
              aggregate_weights=Clipping_Median(old_weights, weights_lis)
              set_parameters(net, aggregate_weights)
              loss, acc = test(net, testloader)
              if np.isnan(loss):
                  old_weights = old_weights
                  set_parameters(net, old_weights)
                  loss, acc = test(net, testloader)
              else:
                  old_weights=aggregate_weights
                  set_parameters(net, old_weights)
              print('global_acc: ', acc, 'loss: ', loss)
              EB_acc.append(acc)
          f = open(norm+mode+str(seed)+"EB.csv", 'w')
          writer = csv.writer(f)
          writer.writerow(EB_acc)
          f.close()



      net = torch.load("mnist_init").to(**setup)


      dummy_id = np.random.choice(len(trainset), 200) #Local data owned by attackers

      true_att_trainset = Subset(trainset, dummy_id)
      valiloader = DataLoader(true_att_trainset, batch_size=200, shuffle=True)
      attloader = DataLoader(true_att_trainset, batch_size = batch_size, shuffle = True)

      print("----------------TD3_Clipping_Median--------------")

      start_time = time.time()
      old_weights = get_parameters(net)
      td3_last_acc = []

      for rnd in range(1000) :
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
          if rnd > 100:
            model_number = min(int((rnd)/5) * 1000, 80000)
            model_path = 'mnist_clipping_median_q0.1/rl_model_'+str(model_number)+'_steps.zip'
            model = TD3.load(model_path)
            action, _ = model.predict(obs)

            #Adjust this to the same parameters in the environments you use
            action[0] = action[0]*14.9+15.0 #epsilon [0,10]
            action[1] = action[1]*24+25  #local step [1:1:50]
          new_weights=[]
          if rnd <= 100:
            for cid in cids:
              set_parameters(net,old_weights)
              train_real(net, trainloaders[cid], epochs=1, lr=lr)
              #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
              new_weight=get_parameters(net)
              new_weights.append(new_weight)
          else:
            for cid in exclude(cids,att_ids):
                set_parameters(net,old_weights)
                train_real(net, trainloaders[cid], epochs=1, lr=lr)
                #train(self.net, self.train_iters[cid%10], epochs=1, lr=lr)
                new_weight=get_parameters(net)
                new_weights.append(new_weight)
            att_weights_lis=[]
            set_parameters(net, old_weights)
            #train(self.net, self.train_iter, epochs=int(action[2]), lr=action[1], mode = False)
            train_real(net, valiloader, epochs=int(action[1]), lr=lr)
            new_weight=get_parameters(net)
            loss, acc = test(net, valiloader)
            #print(self.rnd, loss, acc)
            check = int(action[1])
            while np.isnan(loss):
                check = max(check - 1, 0)
                set_parameters(net, old_weights)
                train_real(net, valiloader, epochs=check, lr=lr)
                new_weight=get_parameters(net)
                loss, acc = test(net, valiloader)
                print(rnd, loss, acc, check)
                if check == 0:
                  new_weight = copy.deepcopy(old_weights)
                  break
            att_weights_lis.append(new_weight)

            for cid in common(cids,att_ids):
                new_weight=craft_att(old_weights, average(att_weights_lis), -1, action[0])
                new_weights.append(new_weight)
          #print(len(new_weights))
          #aggregate_weights = average(weights_lis)
          #aggregate_weights = Median(old_weights, weights_lis)
          #aggregate_weights = Krum(old_weights, new_weights, len(common(cids, att_ids)))
          #aggregate_weights=Clipping(old_weights, weights_lis)
          #aggregate_weights=FLtrust(old_weights, weights_lis, root_iter, lr = lr)
          aggregate_weights=Clipping_Median(old_weights, new_weights)

          set_parameters(net, aggregate_weights)
          loss, acc = test(net, testloader)
          if np.isnan(loss):
            old_weights = old_weights
          else:
            old_weights = aggregate_weights
          set_parameters(net, old_weights)
          epoch_end_time = time.time()
          per_epoch_ptime = epoch_end_time - epoch_start_time
          print ('epoch %d , acc test %.2f%% , loss test %.2f ptime %.2fs .' \
              %(rnd, 100*acc, loss, per_epoch_ptime))
          td3_last_acc.append(acc)
      f = open(norm+mode+str(seed)+"RL.csv", 'w')
      writer = csv.writer(f)
      writer.writerow(td3_last_acc)
      f.close()
