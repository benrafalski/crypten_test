import torch
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

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

num_clients = 5
batch_size = 10
total_train_size = 1000
examples_per_client = total_train_size // num_clients

train_dataset = SepsisDataset(total_train_size)


client_datasets = random_split(train_dataset, [min(i + examples_per_client, 
            total_train_size) - i for i in range(0, total_train_size, examples_per_client)])

client_datasets = [DataLoader(c, batch_size=batch_size, shuffle=True) for c in client_datasets]

client_features = [torch.cat([X for X, y in dataset], dim=0) for dataset in client_datasets]
client_labels = [torch.cat([y for X, y in dataset], dim=0) for dataset in client_datasets]

# features = torch.cat(, dim=0)
# labels = torch.cat(, dim=0)

dir = "client_data_sepsis"

for i in range(len(client_features)):
    torch.save(client_features[i], os.path.join(dir, "features", f"client{i}_features.pth"))

for i in range(len(client_labels)):
    torch.save(client_labels[i], os.path.join(dir, "labels", f"client{i}_labels.pth"))