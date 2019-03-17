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

ap = argparse.ArgumentParser(description='predict-file')

ap.add_argument('input_img', default='paind-project/flowers/test/1/image_06752.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='/home/workspace/paind-project/checkpoint.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = ap.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
path = pa.checkpoint



trainloader, vloader, testloader = main.load_data()


main.load_checkpoint(path)


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


probs = predict(image_path, model, top_num=5)

flower_num = image_path.split('/')[2]
title_ = cat_to_name[flower_num]
img = process_image(image_path)
imshow(img, ax, title = title_);
probs, labs, flowers = predict(image_path, model)




print("It's done")
