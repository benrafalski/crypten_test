import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import crypten
import time
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

CLIENTS = 1000
HIDDENLAYER = 1
EPOCHS = 5

print(f'CLIENTS {CLIENTS}, HIDDEN {HIDDENLAYER}, EPOCHS {EPOCHS}')

crypten.init()

dataset = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

def make_dataset(size):
    indices = np.arange(len(dataset))
    train_indices, test_indices = train_test_split(indices, train_size=size*10, stratify=dataset.targets)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=0, batch_size=10)
    test_loader = DataLoader(test_dataset, shuffle=False, num_workers=0, batch_size=10)

    return train_loader, test_loader

train = []
test = []

for i in range(CLIENTS):
    a, b = make_dataset(6000//CLIENTS)
    train.append(a)
    test.append(b)

class Client(crypten.nn.Module):
    def __init__(self):
        super(Client, self).__init__()

        self.layer1 = crypten.nn.Sequential(
            crypten.nn.Linear(28*28, HIDDENLAYER),
            crypten.nn.ReLU()
        )
        self.layer2 = crypten.nn.Sequential(
            crypten.nn.Linear(HIDDENLAYER, HIDDENLAYER),
            crypten.nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Global(crypten.nn.Module):
    def __init__(self, models):
        super(Global, self).__init__()
        self.models = models
        self.layer3 = crypten.nn.Sequential(
            crypten.nn.Linear(1000, 64),
            crypten.nn.ReLU()
        )
        self.layer4 = crypten.nn.Sequential(
            crypten.nn.Linear(64, 10),
            crypten.nn.LogSoftmax(dim=1)
        )

    def forward(self, c):        
        for i in range(len(c)):
            c[i] = self.models[i](c[i])

        if(self.encrypted):
            x = crypten.cat(c, dim=1)
        else:
            x = torch.cat(c, dim=1)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


clients = []
for i in range(CLIENTS):
    model = Client()
    model.encrypt()
    clients.append(model)


model = Global(clients)
model.encrypt()

loss_criterion = crypten.nn.CrossEntropyLoss()
optimizer = crypten.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)

def enc_data(data):
    X, y = data
    x_enc = crypten.cryptensor(X.view(-1, 784))
    y_one_hot = torch.nn.functional.one_hot(y)
    y_enc = crypten.cryptensor(y_one_hot)
    return x_enc, y_enc

start = time.time()
for epoch in range(EPOCHS): 
    i=0
    for data in zip(*train): 

        X = []
        y = []
        for d in data:
            a, b = enc_data(d)
            X.append(a.view(-1, 784))
            y.append(b)

        output = model(X)  
        if(output.size() != y[0]._tensor.size()):
            continue
        loss = loss_criterion(output, y[0])  
        model.zero_grad() 
        loss.backward()  
        optimizer.step()
        if i%100 == 99:
            print(f'epoch={epoch}, batch={i}')
        i+=1

    
print(f'Runtime : {time.time()-start}')


correct = 0
total = 0
model.eval()

with torch.no_grad():

    for _i in range(CLIENTS):
        clients[_i].decrypt()
    model.decrypt()
    for data in zip(*test):
        X = []
        y = []
        for d in data:
            a, b = d
            X.append(a.view(-1, 784))
            y.append(b) 
        output = model(X)  
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[0][idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 2))

PATH = "models/aggregate_ct.pth"

state = {
    'epoch': 1,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}
torch.save(state, PATH)


# Clients   Runtime   Accuracy  Epochs
# 2         1729.15   0.64      1
# 10        7210.47   0.70      20
# 100       578.86    0.10      5
# 500       1941.17   0.10      5
# 1000      /         ?          

