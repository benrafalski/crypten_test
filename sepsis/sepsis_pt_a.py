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
from statistics import mean
from sepsis_dataset import SepsisDataset, split_data_loaders



class Client(nn.Module):
    def __init__(self, hidden):
        super(Client, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, hidden)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Global(nn.Module):  
    def __init__(self, models):
        super(Global, self).__init__()
        self.models = models
        self.fc3 = nn.Linear(1000, 50)
        self.fc4 = nn.Linear(50, 2)

    def forward(self, c):

        for i in range(len(c)):
            c[i] = self.models[i](c[i])
        x = torch.cat(c, dim=1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

def train(num_epochs, train_sets, model, loss_criterion, optimizer):
    start = time.time()
    for epoch in range(num_epochs):
        print(f'\nEPOCH : {epoch+1}') 
        batch=0
        for data in zip(*train_sets): 

            X = []
            y = []
            for d in data:
                a, b = d
                X.append(a)
                y.append(b)

            output = model(X)  
            
            loss = loss_criterion(output, y[0])  
            model.zero_grad() 
            loss.backward()  
            optimizer.step()
            
            if batch%10 == 9:
                print(f'+', end="")
            batch+=1 
    print(f'\nRuntime : {time.time()-start}')
        



def evaluate(num_clients, model, clients, test):
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for data in zip(*test):
            X = []
            y = []
            for d in data:
                a, b = d
                X.append(a)
                y.append(b) 
            output = model(X)  
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[0][idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 3))
    



def save(path, epochs, client, optim):
    print(f"Saving model to {path}...")
    state = {
        'epoch': epochs,
        'state_dict': client.state_dict(),
        'optimizer': optim.state_dict(),
    }
    torch.save(state, path)


# 10 clients? -> 10 epochs 0.932
# 100 clients? -> 10 epochs 0.950
# 500 clients? -> 10 epochs 0.935 


def main():
    # parameters
    PATH = "models/aggregate_ct.pth"
    CLIENTS = 1000
    HIDDENLAYER = 1000//CLIENTS
    EPOCHS = 10
    TRAIN_SIZE = 100000//CLIENTS
    TEST_SIZE = 100000
    print(f'CLIENTS {CLIENTS}, HIDDEN {HIDDENLAYER}, EPOCHS {EPOCHS}, SIZE {TRAIN_SIZE}')

    # data setup
    sepsis_train = SepsisDataset(TRAIN_SIZE)
    sepsis_test = SepsisDataset(TEST_SIZE)
    trainset = []
    test = []
    for _ in range(CLIENTS):
        a, _ = split_data_loaders(sepsis_train)
        _, b = split_data_loaders(sepsis_test)
        trainset.append(a)
        test.append(b)

    clients = []
    for _ in range(CLIENTS):
        model = Client(HIDDENLAYER)
        clients.append(model)

    model = Global(clients)

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)
    train(EPOCHS, trainset, model, loss_criterion, optimizer)
    evaluate(CLIENTS, model, clients, test)
    save(PATH, EPOCHS, model, optimizer)

if __name__ == "__main__":
    main()



# additive sepsis 
# 10 = 0.952
# 100 = 0.963
# 500 = 0.955
# 1000 = 0.942 


# vertical sepsis 
# 10 = 0.969
# 100 = 0.968
# 500 = 0.966
# 1000 = 0.966 

