import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import crypten
import time
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from statistics import mean
import crypten.mpc as mpc
import crypten.communicator as comm

ALICE = 0
BOB = 1

crypten.init()


x_alice_enc = crypten.load_from_party('tmp/alice_train.pth', src=ALICE)
x_alice_enc = crypten.load_from_party('tmp/alice_train.pth', src=ALICE)
crypten.print(x_alice_enc.size())



