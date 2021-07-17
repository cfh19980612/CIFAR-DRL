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
import queue
import math
import networkx as nx
import argparse
import time
import sys

from tqdm import tqdm, trange
from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Set_dataset(dataset, args):
    if dataset == 'CIFAR10':

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
            trainset, batch_size=args.b, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='/home/ICDCS/cifar-10-batches-py/', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.b, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')

        return trainloader, testloader
    elif dataset == 'MNIST':

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
        return trainloader, testloader
    else:
        print ('Data load error!')
        return 0

def Set_model(client, args):
    print('==> Building model..')
    # Model = [None for i in range (client)]
    # Optimizer = [None for i in range (client)]
    if args.net == 'MNISTNet':
        for i in range (client):
            Model[i] = MNISTNet()
            Optimizer[i] = torch.optim.SGD(Model[i].parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
        global_model = MNISTNet()
        return Model, global_model, Optimizer
    elif args.net == 'MobileNet':
        for i in range (client):
            Model = MobileNet()
            Optimizer = torch.optim.SGD(Model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = MobileNet()
        return Model, global_model, Optimizer
    elif args.net == 'ResNet18':
        for i in range (client):
            Model = ResNet18()
            Optimizer = torch.optim.SGD(Model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = ResNet18()
        return Model, global_model, Optimizer
    elif args.net == 'ResNet50':
        for i in range (client):
            Model = ResNet50()
            Optimizer = torch.optim.SGD(Model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = ResNet50()
        return Model, global_model, Optimizer
    elif args.net == 'ResNet101':
        for i in range (client):
            Model = ResNet101()
            Optimizer = torch.optim.SGD(Model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
        global_model = ResNet101()
        return Model, global_model, Optimizer

def Train(model, optimizer, client, trainloader):
    print('==> Training model..')
    criterion = nn.CrossEntropyLoss().to(device)
    #print(next(model[0].parameters()).is_cuda)
    # cpu ? gpu

    model = model.to(device)
    P = [None for i in range (client)]

    # share a common dataset
    train_loss = 0
    correct = 0
    total = 0
    Loss = 0
    time_start = time.time()
    Batch_time = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
            if batch_idx < 10:
                batch_start = time.time()
                inputs, targets = inputs.to(device), targets.to(device)
                idx = (batch_idx % client)
                model.train()
                optimizer.zero_grad()
                outputs = model(inputs)
                Loss = criterion(outputs, targets)
                Loss.backward()
                optimizer.step()
                train_loss += Loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                batch_end = time.time()
                Batch_time.append(batch_end - batch_start)
            else:
                torch.cuda.empty_cache() 
                break
    ###############################################
    # criterion = nn.CrossEntropyLoss().to(device)
    # #print(next(model[0].parameters()).is_cuda)
    # # cpu ? gpu
    # for i in range(client):
    #     model[i] = model[i].to(device)
    # P = [None for i in range (client)]

    # # share a common dataset
    # train_loss = [0 for i in range (client)]
    # correct = [0 for i in range (client)]
    # total = [0 for i in range (client)]
    # Loss = [0 for i in range (client)]
    # time_start = time.time()
    # Batch_time = []
    # for batch_idx, (inputs, targets) in enumerate(trainloader):
    
    #         if batch_idx < 100:

    #             batch_start = time.time()
    #             inputs, targets = inputs.to(device), targets.to(device)
    #             idx = (batch_idx % client)
    #             model[idx].train()
    #             optimizer[idx].zero_grad()
    #             outputs = model[idx](inputs)
    #             Loss[idx] = criterion(outputs, targets)
    #             Loss[idx].backward()
    #             optimizer[idx].step()
    #             train_loss[idx] += Loss[idx].item()
    #             _, predicted = outputs.max(1)
    #             total[idx] += targets.size(0)
    #             correct[idx] += predicted.eq(targets).sum().item()

    #             batch_end = time.time()
    #             Batch_time.append(batch_end - batch_start)

    # time_end = time.time()

    # if device == 'cuda':
    #     for i in range (client):
    #         model[i].cpu()
    # for i in range (client):
    #     P[i] = copy.deepcopy(model[i].state_dict())

    # return P, (time_end-time_start)

def Test(model, testloader):
    # cpu ? gpu
    model = model.to(device)
    P = model.state_dict()
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        indx_target = target.clone()
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        test_loss += F.cross_entropy(output, target).data
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.cpu().eq(indx_target).sum()
    test_loss = test_loss / len(testloader) # average over number of mini-batch
    accuracy = float(correct / len(testloader.dataset))
    if device == 'cuda':
        model.cpu()
    return accuracy, test_loss.item()

def Aggregate(model, client):
    P = []
    for i in range (client):
        P.append(copy.deepcopy(model[i].state_dict()))
    for key in P[0].keys():
        for i in range (client):
            if i != 0:
                P[0][key] =torch.add(P[0][key], P[i][key])
        P[0][key] = torch.true_divide(P[0][key],client)
    return P[0]


def run(dataset, client, net):
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--b',default=128,type=int,help='batch_size')
    parser.add_argument('--e',default=1,type=int,help='epoch')
    parser.add_argument('--net', default=net,type=str,help='Net')
    args = parser.parse_args()


    X, Y, Z = [], [], []
    trainloader, testloader = Set_dataset(dataset, args)
    model, global_model, optimizer = Set_model(client, args)
    # pbar = tqdm(range(args.epoch))
    start_time = 0
    # for i in pbar:
    return model, optimizer, trainloader, client

    # Temp, process_time = Train(model, optimizer, client, trainloader)
        # for j in range (client):
        #     model[j].load_state_dict(Temp[j])
        # global_model.load_state_dict(Aggregate(copy.deepcopy(model), client))
        # acc, loss = Test(global_model, testloader)
        # pbar.set_description("Epoch: %d Accuracy: %.3f Loss: %.3f Time: %.3f" %(i, acc, loss, start_time))
        # for j in range (client):
        #     model[j].load_state_dict(global_model.state_dict())
        # start_time += process_time
        # X.append(start_time)
        # Y.append(acc)
        # Z.append(loss)
    # location_acc = '/home/cifar-gcn-drl/Test_data/FedAVG_ACC.csv'
    # dataframe_1 = pd.DataFrame(X, columns=['X'])
    # dataframe_1 = pd.concat([dataframe_1, pd.DataFrame(Y,columns=['Y'])],axis=1)
    # dataframe_1.to_csv(location_acc,mode = 'w', header = False,index=False,sep=',')

    # location_loss = '/home/cifar-gcn-drl/Test_data/FedAVG_LOSS.csv'
    # dataframe = pd.DataFrame(X, columns=['X'])
    # dataframe = pd.concat([dataframe, pd.DataFrame(Z,columns=['Z'])],axis=1)
    # dataframe.to_csv(location_loss,mode = 'w', header = False,index=False,sep=',')

if __name__ == '__main__':
    step = 10
    Model = [None for i in range (step)]
    Trainloader = [None for i in range (step)]
    Optimizer = [None for i in range (step)]
    Client = [None for i in range (step)]
    for i in range (step):
        if i%4 == 0: Model[i], Optimizer[i], Trainloader[i], Client[i] = run(dataset = 'CIFAR10', client = 1, net = 'MobileNet')
        elif i%4 == 1: Model[i], Optimizer[i], Trainloader[i], Client[i] = run(dataset = 'CIFAR10', client = 1, net = 'ResNet18')
        elif i%4 == 2: Model[i], Optimizer[i], Trainloader[i], Client[i] = run(dataset = 'CIFAR10', client = 1, net = 'MobileNet')
        elif i%4 == 3: Model[i], Optimizer[i], Trainloader[i], Client[i] = run(dataset = 'CIFAR10', client = 1, net = 'ResNet18')

    for i in range (step):
        if i%4 == 0: 
            Train(Model[i], Optimizer[i], Client[i], Trainloader[i])
            torch.cuda.empty_cache()
        elif i%4 == 1: 
            Train(Model[i], Optimizer[i], Client[i], Trainloader[i])
            torch.cuda.empty_cache()
        elif i%4 == 2: 
            Train(Model[i], Optimizer[i], Client[i], Trainloader[i])
            torch.cuda.empty_cache()
        elif i%4 == 3: 
            Train(Model[i], Optimizer[i], Client[i], Trainloader[i])
            torch.cuda.empty_cache()


