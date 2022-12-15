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

CLIENTS = 500
HIDDEN = 2
EPOCHS = 5
print(f'CLIENTS {CLIENTS}, HIDDEN {HIDDEN}, EPOCHS {EPOCHS}')

torch.multiprocessing.set_sharing_strategy('file_system')

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
    test_loader = DataLoader(test_dataset, shuffle=True, num_workers=0, batch_size=10)

    return train_loader, test_loader

# trainA, testA = make_dataset(3000)
# trainB, testB = make_dataset(3000)

train = []
test = []



for i in range(CLIENTS):
    a, b = make_dataset(6000//CLIENTS)
    train.append(a)
    test.append(b)

class Client(nn.Module):
    def __init__(self, n):
        super(Client, self).__init__()
        self.fc1 = nn.Linear(28*28, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class GlobalModel(nn.Module):
    def __init__(self, models):
        super(GlobalModel, self).__init__()
        self.models = models
        self.fc3 = nn.Linear(1000, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, c):

        for i in range(len(c)):
            c[i] = self.models[i](c[i])
        x = torch.cat(c, dim=1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

models = []
for i in range(CLIENTS):
    m = Client(CLIENTS)
    models.append(m)


model = GlobalModel(models)

loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)

start = time.time()
for epoch in range(EPOCHS):
    print(f'\nEPOCH {epoch+1}') 
    batch=0
    for data in zip(*train): 
        X = []
        y = []
        for d in data:
            a, b = d
            X.append(a.view(-1, 784))
            y.append(b)
        output = model(X)  
        loss = loss_criterion(output, y[0])  
        model.zero_grad() 
        loss.backward()  
        optimizer.step() 
        if batch%10 == 9:
            print(f'+', end="")
        batch+=1 
        # break

    
print(f'Runtime : {time.time()-start}')

correct = 0
total = 0

with torch.no_grad():
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

PATH = "models/aggregate_pt.pth"

state = {
    'epoch': 1,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}
torch.save(state, PATH)
