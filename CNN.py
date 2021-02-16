import os
os.environ['CUDA_ENABLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import random
import argparse
import copy
import pandas as pd
import numpy as np
from utils import progress_bar
from models import *
from multiprocessing import Pool
import queue

import syft as sy  # <-- NEW: import the Pysyft library
hook = sy.TorchHook(torch)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
worker = []
for i in range(10):
    worker.append(sy.VirtualWorker(hook, id="worker"+str(i)))
# worker0 = sy.VirtualWorker(hook, id="worker0")
# worker1 = sy.VirtualWorker(hook, id="worker1")  # <-- NEW: define remote worker bob
# worker2 = sy.VirtualWorker(hook, id="worker2")  # <-- NEW: and alice
# worker3 = sy.VirtualWorker(hook, id="worker3")
# worker4 = sy.VirtualWorker(hook, id="worker4")
# worker5 = sy.VirtualWorker(hook, id="worker5")
# worker6 = sy.VirtualWorker(hook, id="worker6")
# worker7 = sy.VirtualWorker(hook, id="worker7")
# worker8 = sy.VirtualWorker(hook, id="worker8")
# worker9 = sy.VirtualWorker(hook, id="worker9")



class cnn(nn.Module):
    def __init__(self, Client, Dataset, Net):
        self.p = 0.5
        self.dataset = Dataset
        self.net = Net
        self.Model = [None for i in range (Client)]
        self.Optimizer = [None for i in range (Client)]
        # cpu ? gpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'

        self.args, self.trainloader, self.testloader = self.Set_dataset()

    # Preparing data
    def Set_dataset(self):
        if self.dataset == 'CIFAR10':
            parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
            parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
            parser.add_argument('--resume', '-r', action='store_true',
                                help='resume from checkpoint')
            parser.add_argument('--world-size', default=2, type=int,
                    help='number of distributed processes')
            parser.add_argument('--dist-url', default='tcp://163.143.0.120:2222', type=str,
                                help='url used to set up distributed training')
            parser.add_argument('--dist-backend', default='gloo', type=str,
                                help='distributed backend')
            parser.add_argument('--dist-rank', default=0, type=int,
                                help='rank of distributed processes')
            args = parser.parse_args()
            best_acc = 0  # best test accuracy
            start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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
                root='./data/', train=True, download=True, transform=transform_train)
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=128, shuffle=True, num_workers=2)

            # # federated train loader
            # federated_train_loader = sy.FederatedDataLoader(trainset.federated((worker1,worker2,worker3,worker4,worker5,worker6,worker7,worker8,worker9)),
            #     batch_size=64,shuffle=True, num_workers=1)

            testset = torchvision.datasets.CIFAR10(
                root='./data/', train=False, download=True, transform=transform_test)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=100, shuffle=False, num_workers=2)

            classes = ('plane', 'car', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck')

            return args, trainloader, testloader
        elif self.dataset == 'MNIST':
            parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
            parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
            parser.add_argument('--resume', '-r', action='store_true',
                                help='resume from checkpoint')
            parser.add_argument('--world-size', default=2, type=int,
                    help='number of distributed processes')
            parser.add_argument('--dist-url', default='tcp://163.143.0.120:2222', type=str,
                                help='url used to set up distributed training')
            parser.add_argument('--dist-backend', default='gloo', type=str,
                                help='distributed backend')
            parser.add_argument('--dist-rank', default=0, type=int,
                                help='rank of distributed processes')
            args = parser.parse_args()
            best_acc = 0  # best test accuracy
            start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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
            # # federated train loader
            # federated_train_loader = sy.FederatedDataLoader(trainset.federated((worker1,worker2,worker3,worker4,worker5,worker6,worker7,worker8,worker9)),
            #     batch_size=64,shuffle=True, num_workers=1)

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
    # building models
    def Set_Environment(self, Client):
        print('==> Building model..')

        if self.dataset == 'MNIST':
            for i in range (Client):
                self.Model[i] = MNISTNet()
                self.Optimizer[i] = torch.optim.SGD(self.Model[i].parameters(), lr=self.args.lr,
                                momentum=0.9, weight_decay=5e-4)
            global_model = MNISTNet()
            return self.Model, global_model

        elif self.dataset == 'CIFAR10':
            if self.net == 'MobileNet':
                for i in range (Client):
                    self.Model[i] = MobileNet()
                    self.Optimizer[i] = torch.optim.SGD(self.Model[i].parameters(), lr=self.args.lr,
                                momentum=0.9, weight_decay=5e-4)
                global_model = MobileNet()
                return self.Model, global_model

    # CNN training process
    def CNN_train(self, criterion, Client):
        # training
        train_loss = 0
        correct = 0
        total = 0
        Loss = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            worker_idx = batch_idx%Client
            self.Model[worker_idx] = self.Model[worker_idx].to(self.device)
            self.Model[worker_idx].train()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.Optimizer[i].zero_grad()
            outputs = self.Model[worker_idx](inputs)
            Loss = criterion(outputs, targets)
            Loss.backward()
            self.Optimizer[i].step()

            train_loss += Loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if self.device == 'cuda':
                self.Model[worker_idx].cpu()

    # multiple processes to train CNN models
    def CNN_processes(self, epoch, Client):
        P = [None for i in range (Client)]
        # loss func
        criterion = nn.CrossEntropyLoss()

        # share a common dataset
        train_loss = [0 for i in range (Client)]
        correct = [0 for i in range (Client)]
        total = [0 for i in range (Client)]
        Loss = [0 for i in range (Client)]
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                if batch_idx < 390:
                    client = batch_idx % Client
                    self.Model[client].train()
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    self.Optimizer[client].zero_grad()
                    outputs = self.Model[client](inputs)
                    Loss[client] = criterion(outputs, targets)
                    Loss[client].backward()
                    self.Optimizer[client].step()

                    train_loss[client] += Loss[client].item()
                    _, predicted = outputs.max(1)
                    total[client] += targets.size(0)
                    correct[client] += predicted.eq(targets).sum().item()

                    progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                % (train_loss[client]/(batch_idx+1), 100.*correct[client]/total[client], correct[client], total[client]))
        # criterion = nn.CrossEntropyLoss()
        # self.CNN_train(criterion, Client)
        for i in range (Client):
            P[i] = copy.deepcopy(self.Model[i].state_dict())
        return P

    # CNN_test
    def CNN_test(self, epoch, Model):
        Model = Model.to(self.device)
        if self.device == 'cuda':
            Model = torch.nn.DataParallel(Model)

        Model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.testloader:
            indx_target = target.clone()
            if self.device == 'cuda':
                data, target = data.cuda(), target.cuda()
#             with torch.no_grad(data,target):

            output = Model(data)
            test_loss += F.cross_entropy(output, target).data
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.cpu().eq(indx_target).sum()

        test_loss = test_loss / len(self.testloader) # average over number of mini-batch
        accuracy = float(correct / len(self.testloader.dataset))
        if self.device == 'cuda':
            Model.cpu()
        return accuracy

    # local_aggregate
    def Local_agg(self, model, i, Client, Imp, latency):
        # print ('Action: ',p)
        Imp = np.array(Imp).reshape((Client,Client))
        # print ('P: ', p)
        time = 0
        Q = []
        P = copy.deepcopy(model.state_dict())
        for j in range (Client):
            Q.append(copy.deepcopy(self.Model[j].state_dict()))
        for key, value in P.items():
            m = 0
            for j in range (Client):
                if i != j:
                    if Imp[i,j] > 0:
#                     P[key] = P[key] + (Imp[i,j]/Imp[i].sum())*Q[j][key]
                        P[key] = P[key] + Q[j][key]
                        m += 1
            P[key] = torch.true_divide(P[key],m+1)

        for j in range (Client):
            # if self.G.has_edge(i,j):
            time += latency[i][j]
        return P, time

    # Global aggregate
    def Global_agg(self, Client):

        P = copy.deepcopy(self.Model[0].state_dict())
        for key, value in P.items():  
            for i in range (1,Client,1):
                temp = copy.deepcopy(self.Model[i].state_dict())
                P[key] = P[key] + temp[key]
            P[key] = torch.true_divide(P[key],Client)
        return temp

    # step time cost
    def step_time(self, T):
        time = max(T)
        return time

    # to CSV
    def toCsv(self, times, score):
        dataframe = pd.DataFrame(times, columns=['X'])
        dataframe = pd.concat([dataframe, pd.DataFrame(score,columns=['Y'])],axis=1)
        dataframe.to_csv('/home/ICDCS-CIFAR/Test_data/test.csv',mode = 'w', header = False,index=False,sep=',')
    
    # return model
    def toModel(self):
        return self.Model
    

    def forward(self, epoches, Client):
        times, score = [], []
        t = 0
        args, trainloader, testloader = self.Set_dataset()
        self.Set_Environment(args)

        global_model = MobileNet() if self.net == 'MobileNet' else VGG('VGG19')

        # GAT network
        net = GATLayer(self.g,in_dim = 864,out_dim = 20)

        for epoch in range(0, epoches):
            
            Tim, Loss = [], []
            # Loss = [0 for i in range (Client)]

            P = self.CNN_train(epoch,trainloader)

            for i in range (Client):
                self.Model[i].load_state_dict(P[i])

            # global model   
            # global_model = self.Global_agg() 
            
            accuracy = self.CNN_test(epoch,self.Model[0],testloader)

            score.append(accuracy)

            # aggregate local model
            # Step 1: calculate the weight for each neighborhood
            net.update_graph(self.Model, Client)
            net.forward()
            # Step 2: aggregate the model from neighborhood
            for i in range (5):
                P_new = [None for m in range (Client)]
                for x in range (Client):
                    P_new[x], temp = self.Local_agg(self.Model[x],x)
                    Tim.append(temp)
            # update     
            for client in range (Client):
                self.Model[client].load_state_dict(P_new[client])

            times, t = self.step_time(times, Tim, t)

            
            self.toCsv(times,score)

