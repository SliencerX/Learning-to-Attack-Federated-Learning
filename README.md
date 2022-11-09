# Learning-to-Attack-Federated-Learning
This is the reference code for implementing our Learning-to-Attack-Federated-Learning framework. [Paper link](https://openreview.net/pdf?id=4OHRr7gmhd4)

## Code Structure
utilities.py contains all helper functions and defense algorithms including median, clipping median, krum and FLtrust.
exp_environments.py contains all environments used for training attacker policy.
denoise.py contains code for training autoencoder which is used in distribution learning.
sim_train.py contains code for distribution learning and policy learning. 
test.py contains code for testing all other baselines including no attack, inner product manipulation(IPM), explicit boosting(EB), local model poisoning attack(LMP)

## Setup Environment

Please run the following command to install required packages

```
# requirements
pip install -r requirements.txt
```

## Autoencoder
```
#You can change dataset to train your own autoencoder
python3 denoise.py
```

## Distribution Learning and Policy Learning
```
#Default settings in sim_train.py are MNIST, Clipping Median and q = 0.1, you can change these settings in sim_train.py and select coresponding environment from exp_experiments.py
python3 sim_train.py
```
## Test
```
#Change the model dir to your own experiment, you can change dataset and defense method to match your own experiment
python3 test.py
```
## Cite
@inproceedings{
li2022learning,
title={Learning to Attack Federated Learning: A Model-based Reinforcement Learning Attack Framework},
author={Henger Li and Xiaolin Sun and Zizhan Zheng},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=4OHRr7gmhd4}
}

## References
Our inverting gradients implementation is modified from https://github.com/JonasGeiping/invertinggradients

The implementation of FL system is based on https://github.com/adap/flower (Flower - A Friendly Federated Learning Framework)
