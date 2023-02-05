import torch
from torch.utils.data import DataLoader, Dataset
import random
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader(DataLoader):
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            for batch in self.dl:
                yield to_device(batch, self.device)

        def __len__(self):
            return len(self.dl)

class SepsisDataset(Dataset):
    def __init__(self, train, num_clients):
        file_out = pd.read_csv('sepsis/sepsis_data/sepsis_survival_primary_cohort.csv')

        if train:
            x = file_out.iloc[0:num_clients*1000, 0:3].values
            y = file_out.iloc[0:num_clients*1000, 3].values
        else:
            x = file_out.iloc[100000:110203, 0:3].values
            y = file_out.iloc[100000:110203, 3].values

        # feature scaling
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y

        # convert to tensors
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class RandomDataset(Dataset):
    def __init__(self, train):
        random.seed(8560)
        x, y = make_blobs(n_samples=110204, centers=3, n_features=4)

        if train:
            x = x[0:100000]
            y = y[0:100000]
        else:
            x = x[100000:110203]
            y = y[100000:110203]


        # feature scaling
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y

        # convert to tensors
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]