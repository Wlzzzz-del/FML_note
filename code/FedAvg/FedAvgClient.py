import torch
import torch.nn as nn
import torch.functional as F
import os
import json
from torch.utils.data import DataLoader
from userbase import User

class UserAVG(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer):
        super.__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda, local_epochs)

        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr = learning_rate)
    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.parameter):
            for model_grad, new_grad in zip(self.model.parameters(),new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]
    def train(self, epochs):
        ## 这个LOSS是不是出错了。。
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs+1):
            self.model.train()
            X,y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
            self.clone_model_parameter(self.model.parameters(), self.local_model)
        return LOSS