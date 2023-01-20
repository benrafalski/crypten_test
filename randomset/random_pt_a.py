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
from random_dataset import split_data_loaders

class Client(nn.Module):
    def __init__(self, hidden):
        super(Client, self).__init__()
        self.fc1 = nn.Linear(4, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Global(nn.Module):  
    def __init__(self, models):
        super(Global, self).__init__()
        self.models = models
        self.fc3 = nn.Linear(1000, 50)
        self.fc4 = nn.Linear(50, 3)

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
        avg_acc = []
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
            
            with torch.no_grad():
                _, predictions = torch.max(output, dim=1)
                acc = torch.sum(predictions == y[0]).item() / len(predictions)
                avg_acc.append(acc)

        print(f'Epoch accuracy is {round(mean(avg_acc), 5)}')


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

    print("Accuracy: ", round(correct/total, 5))
    



def save(path, epochs, client, optim):
    print(f"Saving model to {path}...")
    state = {
        'epoch': epochs,
        'state_dict': client.state_dict(),
        'optimizer': optim.state_dict(),
    }
    torch.save(state, path)


# 10 clients? -> 5 epochs 0.9999
# 100 clients? -> 20 epochs 0.9996
# 500 clients? -> 50 epochs 0.9990
# 1000 clients? -> 80 epochs 0.4500


def main():
    # parameters
    PATH = "models/random_pt_a.pth"
    CLIENTS = 1000
    HIDDENLAYER = 1000//CLIENTS
    # HIDDENLAYER = 1
    EPOCHS = 50
    TRAIN_SIZE = 100000//CLIENTS
    # TRAIN_SIZE = 200
    print(f'CLIENTS {CLIENTS}, HIDDEN {HIDDENLAYER}, EPOCHS {EPOCHS}, SIZE {TRAIN_SIZE}')

    # data setup
    trainset = []
    test = []
    for _ in range(CLIENTS):
        a, b = split_data_loaders(TRAIN_SIZE)
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




# accuracy

# 10 = 0.9969
# 100 = 0.9957
# 500 = 0.3331
# 1000 = 0.3326
