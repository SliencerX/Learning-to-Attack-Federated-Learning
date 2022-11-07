# Learning-to-Attack-Federated-Learning
This is the reference code for implementing our Learning-to-Attack-Federated-Learning framework.

## setup environment

Please run the following command to install required packages

```
# requirements
pip install -r requirements.txt
```
## Distribution Learning and Policy Learning
```
#Default settings in sim_train.py are MNIST, Clipping Median and q = 0.1, you can change these settings in sim_train.py and select coresponding environment from exp_experiments.py
python3 sim_train.py
```
## Test
```
#Change the model dir to your own experiment
python3 test.py
```
## Reference
Our inverting gradients implementation is modified from https://github.com/JonasGeiping/invertinggradients
