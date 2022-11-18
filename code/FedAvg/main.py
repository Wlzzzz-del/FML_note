from FedAvgClient import UserAVG
from serverAvg import FedAvg
import torch
import numpy as np
from models import Net


_gpu = 0
_model = "cnn"
_dataset = "Minst"
_batch_size = 20
_learning_rate = 0.005
_beta = 1.0
_lamda = 15
_num_global_iters = 800
_local_epochs = 20
_optimizer = "SGD"
_numusers = 20
_times = 5
_device = torch.device("cuda:{}".format(_gpu)if torch.cuda.is_available() and _gpu!= -1 else "cpu")
model = Net().to(_device),"cnn"

for i in range(_times):
    print("-------------------runing tims:------------------")
    server = FedAvg(_device,_dataset,"FedAvg",_model,_batch_size,_learning_rate,_beta,_lamda,_num_global_iters,_local_epochs,_optimizer,_numusers,_times)
    server.train()
    server.test()
