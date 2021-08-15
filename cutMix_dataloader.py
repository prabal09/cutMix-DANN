# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:55:28 2021

@author: prabal
"""

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet as RN
import pyramidnet as PYRM
import utils
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import warnings
import matplotlib.pyplot as plt
import numpy as np
import PIL
warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--net_type', default='pyramidnet', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='cifar10', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)



def dispImg(inp,idx):
    plt.imshow(np.transpose(inp[idx].numpy(), (1, 2, 0)))


def imshow(img):
    imgC = img.clone()
    imgC[0] = imgC[0]*0.24705882352941178  + 0.4913725490196078
    imgC[1] = imgC[1]*0.24352941176470588  + 0.4823529411764706
    imgC[2] = imgC[2]*0.2615686274509804  + 0.4466666666666667
    npimg = imgC.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def imshow2(img):
    imgC = img.clone()
    imgC = imgC/2 + 0.5
    npimg = imgC.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()  
    
#normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
#                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

normalize = transforms.Normalize(mean=[0.5],std = [0.5])

#transform_train = transforms.Compose([
#    transforms.RandomCrop(32, padding=4),
#    transforms.RandomHorizontalFlip(),
#    transforms.ToTensor(),
#    normalize,
#])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
])    

transform_src = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    normalize,
])

    
transform_tgt = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    normalize,
])
def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutMix_loader(source_loader,target_loader):
    X = np.empty((0,3,32,32));y = np.empty((0,));
    # number of samples x number of channels x height x width
    for item1, item2 in zip(source_loader, target_loader):
        image_batch1, labels1 = item1
        image_batch2, labels2 = item2
#        break
        l_np,labels = cutMix(image_batch1,image_batch2)
    X = np.append(X,l_np,axis=0)
    y = np.append(y,labels,axis=0)
    dataset = TensorDataset( Tensor(X), Tensor(y) )
        # Create a data loader from the dataset
        # Type of sampling and batch size are specified at this step
    loader = DataLoader(dataset, batch_size= 16)
    # mix images and lambdas
    #return batch of images and list of lamda values
    return loader



def cutMix(image_batch1,image_batch2):
    l=[];labels = []
    for img1,img2 in zip(image_batch1,image_batch2):
#        break
#        imshow(img1)
#        imshow(img2)
        size = img1.shape
        lam = np.random.beta(1,1)
        bbx1, bby1, bbx2, bby2 = rand_bbox(size,lam)
        img__ = img1.clone()
        img__[0][bbx1:bbx2, bby1:bby2] = img2[0][bbx1:bbx2, bby1:bby2]
        img__[1][bbx1:bbx2, bby1:bby2] = img2[1][bbx1:bbx2, bby1:bby2]
        img__[2][bbx1:bbx2, bby1:bby2] = img2[2][bbx1:bbx2, bby1:bby2]
#        imshow(img__)
        l.append(img__.numpy())
        labels.append(lam)
    l_np = np.asarray(l)
    labels = np.asarray(labels)
    return l_np,labels

def cutMixLabels(labels1,labels2,lam):
    labels = lam*labels1+(1-lam)*labels2
    return labels  


if __name__ =="__main__":
    args = parser.parse_args()
    
    source_loader = torch.utils.data.DataLoader(
        datasets.SVHN('../data', split='train', download=True,transform=transform_src),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    target_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('../data', train=True, download=True, transform=transform_tgt),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    cutmix_loader = cutMix_loader2(source_loader,target_loader)
    
    for inp,tgt in cutmix_loader:
        for i in range(len(inp)):
            imshow2(inp[i])
            print(tgt[i])
        break