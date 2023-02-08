from collections import OrderedDict
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch.nn as nn
import torch.nn.functional as F
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm 
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torchmetrics.classification import BinaryAccuracy
import sys
import time
from statistics import mean
import plotext as plt
import logging

class FederatedNet(nn.Module):    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 5)
        self.fc4 = nn.Linear(5, 1)
        self.track_layers = {'fc1': self.fc1, 'fc2': self.fc2, 'fc3': self.fc3, 'fc4': self.fc4}

    def forward(self, x_batch):
        x = F.relu(self.fc1(x_batch))
        x = F.relu(self.fc2(x))
        return x

    def forward_server(self, x):
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x).view(-1, 10))
        # return torch.softmax(self.fc4(x), dim=1)

    def forward_testing(self, x_batch):
        x = F.relu(self.fc1(x_batch))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x).view(-1, 10))

    # return the dictionary of the layers
    def get_track_layers(self):
        return self.track_layers
    
    # sets each layer's weight and bias to the weight and bias given as argument
    def apply_parameters(self, parameters_dict):
        with torch.no_grad():
            for layer_name in parameters_dict:
                self.track_layers[layer_name].weight.data *= 0
                self.track_layers[layer_name].bias.data *= 0
                self.track_layers[layer_name].weight.data += parameters_dict[layer_name]['weight']
                self.track_layers[layer_name].bias.data += parameters_dict[layer_name]['bias']

    def apply_parameters_server(self, parameters_dict):
        with torch.no_grad():
            layers = ['fc1', 'fc2']
            for layer_name in layers:
                self.track_layers[layer_name].weight.data *= 0
                self.track_layers[layer_name].bias.data *= 0
                self.track_layers[layer_name].weight.data += parameters_dict[layer_name]['weight'].get_plain_text()
                self.track_layers[layer_name].bias.data += parameters_dict[layer_name]['bias'].get_plain_text()
    
    
    # returns a parameter dictionary for each layer
    # dictionary is of the form {"weight": w, "bias": b}
    def get_parameters(self):
        parameters_dict = dict()
        for layer_name in self.track_layers:
            parameters_dict[layer_name] = {
                'weight': self.track_layers[layer_name].weight.data, 
                'bias': self.track_layers[layer_name].bias.data
            }
        return parameters_dict

    def evaluate(self, dataset):
        losses = []
        accs = []
        loss_criterion = torch.nn.BCELoss()
        metric = BinaryAccuracy()
        with torch.no_grad():
            for batch in dataset:
                X, y = batch
                server_out = self.forward_testing(X)
                y = torch.unsqueeze(y, 0)
                loss = loss_criterion(server_out, y)
                with torch.no_grad():
                    acc = metric(server_out, y)
                losses.append(loss)
                accs.append(acc)
        avg_loss = torch.stack(losses).mean().item()
        avg_acc = torch.stack(accs).mean().item()
        return (avg_loss, avg_acc)

class ClientDataset(Dataset):
    def __init__(self, client_num, file_out, data_per_client):
        self.file_out = file_out
        x = self.file_out.iloc[client_num*data_per_client:(client_num+1)*data_per_client, 0:3].values
        y = self.file_out.iloc[client_num*data_per_client:(client_num+1)*data_per_client, 3].values

        # feature scaling
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y

        # convert to tensors
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

class FederatedNetCrypten(crypten.nn.Module):    
    def __init__(self):
        super().__init__()
        self.fc1 = crypten.nn.Linear(3, 5)
        self.relu1 = crypten.nn.ReLU()
        self.fc2 = crypten.nn.Linear(5, 5)
        self.relu2 = crypten.nn.ReLU()
        self.fc3 = crypten.nn.Linear(5, 5)
        self.relu3 = crypten.nn.ReLU()
        self.fc4 = crypten.nn.Linear(5, 1)
        self.sigmoid = crypten.nn.Sigmoid()
        self.track_layers = {'fc1': self.fc1, 'fc2': self.fc2, 'fc3': self.fc3, 'fc4': self.fc4}


    def forward(self, x_batch):
        # x = self.relu1(self.fc1(x_batch))
        # # x = self.relu2(self.fc2(x))
        # return self.relu2(self.fc2(x))
        # x = self.relu3(self.fc3(x))
        # return self.sigmoid(self.fc4(x))
        x = self.relu1(self.fc1(x_batch))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.sigmoid(self.fc4(x).view(-1, 10))

    def forward_server(self, x):
        x = self.relu3(self.fc3(x))
        return self.sigmoid(self.fc4(x).view(-1, 10))

    def forward_testing(self, x_batch):
        x = self.relu1(self.fc1(x_batch))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        return self.sigmoid(self.fc4(x).view(-1, 10))

    # return the dictionary of the layers
    def get_track_layers(self):
        return self.track_layers
    
    # sets each layer's weight and bias to the weight and bias given as argument
    def apply_parameters(self, parameters_dict):
        with torch.no_grad():
            for layer_name in parameters_dict:
                self.track_layers[layer_name].weight.data *= 0
                self.track_layers[layer_name].bias.data *= 0
                self.track_layers[layer_name].weight.data += parameters_dict[layer_name]['weight']
                self.track_layers[layer_name].bias.data += parameters_dict[layer_name]['bias']

    def apply_parameters_server(self, parameters_dict):
        with torch.no_grad():
            layers = ['fc1', 'fc2']
            for layer_name in layers:
                self.track_layers[layer_name].weight.data *= 0
                self.track_layers[layer_name].bias.data *= 0
                self.track_layers[layer_name].weight.data += parameters_dict[layer_name]['weight'].get_plain_text()
                self.track_layers[layer_name].bias.data += parameters_dict[layer_name]['bias'].get_plain_text()
    
    
    # returns a parameter dictionary for each layer
    # dictionary is of the form {"weight": w, "bias": b}
    def get_parameters(self):
        parameters_dict = dict()
        for layer_name in self.track_layers:
            parameters_dict[layer_name] = {
                'weight': self.track_layers[layer_name].weight.data, 
                'bias': self.track_layers[layer_name].bias.data
            }
        return parameters_dict

    def evaluate(self, dataset):
        losses = []
        accs = []
        loss_criterion = torch.nn.BCELoss()
        metric = BinaryAccuracy()
        with torch.no_grad():
            for batch in dataset:
                X, y = batch
                server_out = self.forward_testing(X)
                y = torch.unsqueeze(y, 0)
                loss = loss_criterion(server_out, y)
                with torch.no_grad():
                    acc = metric(server_out, y)
                losses.append(loss)
                accs.append(acc)
        avg_loss = torch.stack(losses).mean().item()
        avg_acc = torch.stack(accs).mean().item()
        return (avg_loss, avg_acc)



crypten.init()
torch.set_num_threads(1)


crypten.common.serial.register_safe_class(FederatedNet)

net = crypten.nn.from_pytorch(FederatedNet(), torch.empty(5,3))
net.encrypt()
net.train()


num_clients = 1
epochs = 1
data_per_client = 100


file_out = pd.read_csv('sepsis/sepsis_data/sepsis_survival_primary_cohort.csv')
file_out.sample(frac=1)

# test_dataset = TestingDataset()
# test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

client_datasets = [DataLoader(ClientDataset(i, file_out, data_per_client), batch_size=10, shuffle=True) for i in range(num_clients)]
# clients = [FederatedNet() for _ in range(num_clients)]

data = DataLoader(ClientDataset(0, file_out, data_per_client), batch_size=10, shuffle=True)

loss_criterion = crypten.nn.MSELoss()
optimizer = crypten.optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)




for epoch in range(epochs):

    crypten.print(f'Starting epoch {epoch+1}')

    avg_acc = []
    for X, y in data:
        X = crypten.cryptensor(X)
        y = torch.unsqueeze(y, 0)
        y = crypten.cryptensor(y)

        output = net.forward_testing(X)
        loss_value = loss_criterion(output, y)








    #     # output = net.forward(X)
    #     # server_output = global_net.forward_server(output)

    #     server_output = net(X)


    #     # print(f'output: {server_output}, label: {y}')
        

    #     loss = loss_criterion(server_output, y)  
    #     net.zero_grad() 
    #     loss.backward()  
    #     optimizer.step()


    #     net.decrypt()

    #     print(f'after {net.get_parameters()}')
    #     a = list(net.parameters())[0].clone()

    #     net.encrypt()


    #     metric = BinaryAccuracy()
    #     with torch.no_grad():
    #         acc = metric(server_output.get_plain_text(), y.get_plain_text())
    #         avg_acc.append(acc.item())

    #     print(torch.equal(a.data, b.data))


    # print(mean(avg_acc))

    #     # crypten.print(f'Average batch accuracy: {mean(avg_acc)}')

        

