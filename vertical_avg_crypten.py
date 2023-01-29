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


class TestingDataset(Dataset):
    def __init__(self):
        file_out = pd.read_csv('sepsis/sepsis_data/sepsis_survival_study_cohort.csv')
        x = file_out.iloc[1:19001, 0:3].values
        y = file_out.iloc[1:19001, 3].values

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

class Client(nn.Module):
    def __init__(self, hidden):
        super(Client, self).__init__()
        self.fc1 = nn.Linear(3, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.track_layers = {'fc1': self.fc1, 'fc2': self.fc2}

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def get_parameters(self):
        parameters_dict = dict()
        for layer_name in self.track_layers:
            parameters_dict[layer_name] = {
                'weight': self.track_layers[layer_name].weight.data, 
                'bias': self.track_layers[layer_name].bias.data
            }
        return parameters_dict


class Global(nn.Module):  
    def __init__(self):
        super(Global, self).__init__()
        self.fc3 = nn.Linear(1000, 50)
        self.fc4 = nn.Linear(50, 2)
        self.track_layers = {'fc3': self.fc3, 'fc4': self.fc4}

    def forward(self, x):
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

    def get_track_layers(self):
        return self.track_layers
    
    def get_parameters(self):
        parameters_dict = dict()
        for layer_name in self.track_layers:
            parameters_dict[layer_name] = {
                'weight': self.track_layers[layer_name].weight.data, 
                'bias': self.track_layers[layer_name].bias.data
            }
        return parameters_dict


crypten.init()
torch.set_num_threads(1)

num_clients = 2
data_per_client = 100

file_out = pd.read_csv('sepsis/sepsis_data/sepsis_survival_primary_cohort.csv')
file_out.sample(frac=1)

test_dataset = TestingDataset()
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

client_datasets = [DataLoader(ClientDataset(i, file_out, data_per_client)) for i in range(num_clients)]

clients = [Client(1000//num_clients) for _ in range(num_clients)]

global_model = Global()

@mpc.run_multiprocess(world_size=num_clients)
def secret_share():
    for dataset in zip(*client_datasets):
        X = [a for a, _ in dataset]
        y = [a for _, a in dataset]

        output = [clients[i](X[i]) for i in range(num_clients)]
        x = torch.cat(output, dim=1)


        crypten.print(x)

        break

        server_output = global_model(x)



secret_share()












