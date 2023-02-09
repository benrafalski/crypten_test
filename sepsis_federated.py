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
    
class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = torch.nn.Linear(3, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x).view(-1, 10))
        return x

batch_size = 10
train_dataset = SepsisDataset(100000)
net = Network()
loss_criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)

num_clients = 5
total_train_size = len(train_dataset)
examples_per_client = total_train_size // num_clients

client_datasets = random_split(train_dataset, [min(i + examples_per_client, 
            total_train_size) - i for i in range(0, total_train_size, examples_per_client)])

client_datasets = [DataLoader(c, batch_size=batch_size, shuffle=True) for c in client_datasets]


features = torch.cat([torch.cat([X for X, y in dataset], dim=0) for dataset in client_datasets], dim=0)
labels = torch.cat([torch.cat([y for X, y in dataset], dim=0) for dataset in client_datasets], dim=0)


num_features = features.size()[0]



epoch_accuracies = []
for epoch in range(5):
    print(f'\nEPOCH : {epoch+1}')
    acc = []
    batch_num = 0
    for i in range(0, num_features, batch_size):
        x_batch = features[i:(i+batch_size)]
        y_batch = labels[i:(i+batch_size)]

        batch_num += 1

        if batch_num % ((num_features//10)//10) == 0:
            print(f'\tStarting batch {batch_num} of {num_features//10}')

        output = net(x_batch)
        y = torch.unsqueeze(y_batch, 0)
        loss = loss_criterion(output, y)  
        net.zero_grad() 
        loss.backward()  
        optimizer.step()

        metric = BinaryAccuracy()
        with torch.no_grad():
            avg_acc = metric(output, y)

        # avg_acc = compute_accuracy(output, y)

        acc.append(avg_acc)

    epoch_acc = torch.stack(acc).mean().item()
    epoch_accuracies.append(epoch_acc)
    print(f'Epoch accuracy: {epoch_acc}')




