import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from statistics import mean
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import BinaryAccuracy
import plotext as plt
import crypten 



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
        x = torch.sigmoid(self.fc4(x))
        return x
    

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
    
def split_data_loaders(data):
    train_dataset, test_dataset = random_split(data, [int(len(data) * 0.9), int(len(data) * 0.1)])
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    return (train_loader, test_loader)


crypten.init()
train, test = split_data_loaders(SepsisDataset(1000))
net = Network()
net = crypten.nn.from_pytorch(net, torch.empty(64, 3))
net.encrypt()
net.train()

loss_criterion = crypten.nn.BCELoss()


epoch_accuracies = []
for epoch in range(50):
    print(f'\nEPOCH : {epoch+1}')
    acc = []
    batch_num = 0
    for X, y in train:

        batch_num += 1

        if batch_num % (len(train)//10) == 0:
            print(f'\tStarting batch {batch_num} of {len(train)}')

        X = crypten.cryptensor(X)
        y = torch.unsqueeze(y, 0)
        y_enc = crypten.cryptensor(y) 


        output = net(X)
        output = output.view(-1, 10)
        
        loss = loss_criterion(output, y_enc)  
        net.zero_grad() 
        loss.backward()  
        net.update_parameters(0.001)


        metric = BinaryAccuracy()
        with torch.no_grad():
            avg_acc = metric(output.get_plain_text(), y)
        
        acc.append(avg_acc)

    epoch_acc = torch.stack(acc).mean().item()
    epoch_accuracies.append(epoch_acc)
    print(f'Epoch accuracy: {epoch_acc}')

