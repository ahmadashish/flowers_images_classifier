import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

import main

ap = argparse.ArgumentParser(description='train.py')
# Command Line ardguments

ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")

pa = ap.parse_args()
where = pa.data_dir



trainloader, validationloader, testloader = main.load_data(where)


model, optimizer, criterion = main.nn_setup(structure,features,Dropout,lr)


main.train_network


main.save_checkpoint(checkpoint, 'checkpoint.pth')


print("it's Done.")
