import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import random
import copy
import pandas as pd
import numpy as np
from utils import progress_bar
import queue
import math
import networkx as nx
import numpy as np
import argparse
import time
from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Set_dataset(dataset):
    if dataset == 'CIFAR10':
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        parser.add_argument('--resume', '-r', action='store_true',
                            help='resume from checkpoint')
        parser.add_argument('--epoch',default=100,type=int,help='epoch')
        args = parser.parse_args()

        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='/home/ICDCS/cifar-10-batches-py/', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='/home/ICDCS/cifar-10-batches-py/', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')

        return args, trainloader, testloader
    elif dataset == 'MNIST':
        parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        parser.add_argument('--resume', '-r', action='store_true',
                            help='resume from checkpoint')
        parser.add_argument('--epoch',default=100,type=int,help='epoch')
        args = parser.parse_args()

        # Data
        print('==> Preparing data..')
        # normalize
        transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
        # download dataset
        trainset = torchvision.datasets.MNIST(root = "./data/",
                        transform=transform,
                        train = True,
                        download = True)
        # load dataset with batch=64
        trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                            batch_size = 64,
                                            shuffle = True)

        testset = torchvision.datasets.MNIST(root="./data/",
                        transform = transform,
                        train = False)

        testloader = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size = 64,
                                            shuffle = False)
        return args, trainloader, testloader
    else:
        print ('Data load error!')
        return 0

def Set_model(net, client, args):
    print('==> Building model..')
    Model = [None for i in range (Client)]
    Optimizer = [None for i in range (Client)]
    if net == 'MNISTNet':
        for i in range (Client):
            Model[i] = MNISTNet()
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        global_model = MNISTNet()
        return Model, global_model, Optimizer
    elif net == 'MobileNet':
        for i in range (Client):
            Model[i] = MobileNet()
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = MobileNet()
        return Model, global_model, Optimizer

def Train(model, client, trainloader):
    criterion = nn.CrossEntropyLoss().to(device)
    # cpu ? gpu
    for i in range(Client):
        model[i] = model[i].to(device)
    P = [None for i in range (Client)]

    # share a common dataset
    train_loss = [0 for i in range (Client)]
    correct = [0 for i in range (Client)]
    total = [0 for i in range (Client)]
    Loss = [0 for i in range (Client)]
    time_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
            if batch_idx < 360:
                client = (batch_idx % Client)
                model[client].train()
                inputs, targets = inputs.to(device), targets.to(device)
                Optimizer[client].zero_grad()
                outputs = model[client](inputs)
                Loss[client] = criterion(outputs, targets)
                Loss[client].backward()
                Optimizer[client].step()

                train_loss[client] += Loss[client].item()
                _, predicted = outputs.max(1)
                total[client] += targets.size(0)
                correct[client] += predicted.eq(targets).sum().item()
    time_end = time.time()
    if self.device == 'cuda':
        for i in range (Client):
            model[i].cpu()
    for i in range (Client):
        P[i] = copy.deepcopy(Model[i].state_dict())

    return P, (time_end-time_start)

def Test(model, testloader):
    # cpu ? gpu
    model = model.to(device)

    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        indx_target = target.clone()
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.cross_entropy(output, target).data
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.cpu().eq(indx_target).sum()
    test_loss = test_loss / len(testloader) # average over number of mini-batch
    accuracy = float(correct / len(testloader.dataset))
    if device == 'cuda':
        model.cpu()
    return accuracy, test_loss

def Aggregate(model, client):
    P = copy.deepcopy(model[0].state_dict())
    for key, value in P.items():
        for i in range (1,Client,1):
            temp = copy.deepcopy(model[i].state_dict())
            P[key] = P[key] + temp[key]
        P[key] = torch.true_divide(P[key],client)
    return P

def run(dataset, net, client):
    args, trainloader, testloader = Set_dataset(dataset)
    model, global_model, optimizer = Set_model(net, client, args)
    pbar = tqdm(range(args.epoch)
    X,Y,Z = [], [], []
    start_time = 0
    for i in range (args.epoch):
        Temp, process_time = Train(model, trainloader)
        for i in range (client):
            model[i].load_state_dict(Temp[i])
        global_model.load_state_dict(Aggregate(model, client))
        acc, loss = Test(global_model, client)
        pbar.set_description("Epoch: %d Accuracy: %.3f Loss: %.3f" %(i, acc, loss))
        start_time += process_time
        X.append(start_time)
        Y.append(acc)
        Z.append(loss)
    location = '/home/cifar-gcn-drl/Test_data/FedAVG.csv'
    dataframe = pd.DataFrame(times, columns=['X'])
    dataframe = pd.concat([dataframe, pd.DataFrame(score,columns=['Y'])],axis=1)
    dataframe = pd.concat([dataframe, pd.DataFrame(loss,columns=['Z'])],axis=1)
    dataframe.to_csv(location,mode = 'w', header = False,index=False,sep=',')

if __name__ == '__main__':
    run(dataset = 'CIFAR10', net = 'MobileNet', client = 10)
