import torch
import os
import numpy as np
import h5py
import copy

class Server:
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda,num_glob_iters, local_epochs, optimizer, num_users, times):

        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.beta = beta
        self.lamda = lamda
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], []
        self.tims =times
    def aggregate_grads(self):
        assert(self.users is not None and len(self.users)> 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            # 带权重系数的梯度
            self.add_grad(user, user.train_samples/ self.total_train_samples)
    def add_grad(self, user, ratio):
        # ratio是系数
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad+ user_grad[idx].clone()*ratio
    
    def send_parameters(self):
        assert(self.users is not None and len(self.users)> 0)
        for user in self.users:
            user.set_parameters(self.model)

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.getparameters()):
            server_param.data = server_param.data + user_param.data.clone()
    
    def aggregate_parameters(self):
        assert (self.selected_users is not None and len(self.selected_users)>0)
        for param in self.model.parameters():
            param.data =torch.zeros_like(param.data)
        total_train = 0
        # 统计用于训练的样本总量
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user,user.train_samples/total_train)
    def save_model(self):
        model_path = os.path.join("model", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server"+".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server"+".pt")
        assert(os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models",self.dataset, "server"+".pt"))

    def select_users(self, round, num_users):
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users
        num_users = min(num_users, len(self.users))
        # 从self.users中挑选num_users个Client参与训练
        return np.random.choice(self.users, num_users, replace= False)
    
    def save_results(self):
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "_" + str(self.learning_rate)+"_"+str(self.beta)+"_"+str(self.lamda)+"_"+str(self.num_users)+"u"+"_"+str(self.batch_size)+"b"+"_"+str(self.local_epochs)
        alg = alg + "_" +str(self.times)
        if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.close()

    def evaluate(self):
        stats = self.test()
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Global Accurancy: ", glob_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ",train_loss)