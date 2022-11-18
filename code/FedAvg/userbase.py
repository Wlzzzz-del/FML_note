import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy

class User:
    """
    联邦学习内用户的基类
    """
    def __init__(self,device, id, train_data, test_data, model, batch_size= 0, learning_rate=0, beta=0, lamda=0, local_epochs = 0):

        self.device= device
        self.model = copy.deepcopy(model)
        self.id = id
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.lamda = lamda
        self.local_epochs = local_epochs
        self.trainloader = DataLoader(train_data, self.batch_size)
        self.testloader = DataLoader(test_data, self.batch_size)
        self.trainloaderfull= DataLoader(train_data, self.train_samples)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        # iter返回一个迭代器集合，通过iter.next调用下一个迭代器
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

    def get_all_train_data(self):
        pass

    def get_all_test_data(self):
        pass

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()
            # local_param.data = new_para.data.clone()

    def get_parameters(self, model):
        for param in self.model.parameters():
            param.detach()# 去掉梯度信息
        return self.model.parameters()

    def clone_model_parameter(self, param, clone_param):
        for _param, c_param in zip(param, clone_param):
            c_param.data = _param.data.clone()
        return clone_param

    def get_grads(self):
        grads=[]
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self):
        self.model.eval()
        test_acc =0
        for x_y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1)==y)).item()
        return test_acc, y.shape[0]

    def train_error_and_loss(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            train_acc +=(torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
        return train_acc, loss, self.train_samples
    
    def get_next_train_batch(self):
        try:
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X.to(self.device), y.to(self.device))
    
    def get_next_test_batch(self):
        try:
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            self.iter_testloader = iter(self.iter_testloader)
            (X, y) = next(self.iter_testloader)
        return (X.to(self.device), y.to(self.device))
    
    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model  = torch.load(os.path.join(model_path, "server"+".pt"))

    @staticmethod
    def model_exitsts():
        return os.path.exists(os.path.join("models", "server"+".pt"))