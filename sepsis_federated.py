import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from statistics import mean
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import BinaryAccuracy
import plotext as plt
# import crypten 

class SepsisDataset(Dataset):
    def __init__(self, data_size):
        file_out = pd.read_csv('sepsis/sepsis_data/sepsis_survival_primary_cohort.csv')
        x = file_out.iloc[1:(data_size+1), 0:3].values
        y = file_out.iloc[1:(data_size+1), 3].values

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

train_dataset = SepsisDataset(1000)

num_clients = 5
total_train_size = len(train_dataset)
examples_per_client = total_train_size // num_clients

client_datasets = random_split(train_dataset, [min(i + examples_per_client, 
            total_train_size) - i for i in range(0, total_train_size, examples_per_client)])

client_datasets = [DataLoader(c, batch_size=10, shuffle=True) for c in client_datasets]




