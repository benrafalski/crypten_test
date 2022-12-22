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

# clients = 10 -> 40 epochs has 0.77 accuracy
# clients = 100 -> 5 epochs has 0.11 accuracy
# clients = 500 -> 5 epochs has 0.11 accuracy
# clients = 1000 -> 5 epochs has 0.11 accuracy

CLIENTS = 1000
HIDDEN = 1000//CLIENTS
EPOCHS = 5
print(f'CLIENTS {CLIENTS}, HIDDEN {HIDDEN}, EPOCHS {EPOCHS}')

torch.multiprocessing.set_sharing_strategy('file_system')

dataset = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))


def make_dataset(size):
    indices = np.arange(len(dataset))
    train_indices, test_indices = train_test_split(indices, train_size=size*10, test_size=10000, stratify=dataset.targets)

    # print(len(train_indices))
    # print(len(test_indices))

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=0, batch_size=10)
    test_loader = DataLoader(test_dataset, shuffle=True, num_workers=0, batch_size=10)

    return train_loader, test_loader

# trainA, testA = make_dataset(3000)
# trainB, testB = make_dataset(3000)

train = []
test = []


# a, b = make_dataset(6000//CLIENTS)


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

    
print(f'\nRuntime : {time.time()-start}')

correct = 0
total = 0
model.eval()

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
        # print(time.time() - s)

print("Accuracy: ", round(correct/total, 3))

PATH = "models/aggregate_pt.pth"

state = {
    'epoch': 1,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}
torch.save(state, PATH)


# CLIENTS 10, HIDDEN 100, EPOCHS 1
# Runtime : 4.476203680038452
# Accuracy:  0.15

# CLIENTS 10, HIDDEN 100, EPOCHS 2
# Runtime : 8.450185775756836
# Accuracy:  0.28

# CLIENTS 10, HIDDEN 100, EPOCHS 3
# Runtime : 12.775931358337402
# Accuracy:  0.35

# CLIENTS 10, HIDDEN 100, EPOCHS 4
# Runtime : 18.12265634536743
# Accuracy:  0.32

# CLIENTS 10, HIDDEN 100, EPOCHS 5
# Runtime : 21.20010232925415
# Accuracy:  0.51

# CLIENTS 10, HIDDEN 100, EPOCHS 6
# Runtime : 25.611743211746216
# Accuracy:  0.45

# CLIENTS 10, HIDDEN 100, EPOCHS 7
# Runtime : 30.067214488983154
# Accuracy:  0.62

# CLIENTS 10, HIDDEN 100, EPOCHS 8
# Runtime : 33.94936728477478
# Accuracy:  0.57

# CLIENTS 10, HIDDEN 100, EPOCHS 9
# Runtime : 39.33009910583496
# Accuracy:  0.63

# CLIENTS 10, HIDDEN 100, EPOCHS 10
# Runtime : 42.78562831878662
# Accuracy:  0.6

# CLIENTS 10, HIDDEN 100, EPOCHS 15
# Runtime : 64.37325668334961
# Accuracy:  0.71

# CLIENTS 10, HIDDEN 100, EPOCHS 20
# Runtime : 86.92861557006836
# Accuracy:  0.71


# CLIENTS 10, HIDDEN 100, EPOCHS 20
# Runtime : 86.92861557006836
# Accuracy:  0.71

# CLIENTS 10, HIDDEN 100, EPOCHS 25
# Runtime : 107.03025197982788
# Accuracy:  0.73

# CLIENTS 10, HIDDEN 100, EPOCHS 30
# Runtime : 127.78702235221863
# Accuracy:  0.75

# CLIENTS 10, HIDDEN 100, EPOCHS 35
# Runtime : 148.59967708587646
# Accuracy:  0.76

# CLIENTS 10, HIDDEN 100, EPOCHS 40
# Runtime : 171.53010773658752
# Accuracy:  0.77

# CLIENTS 10, HIDDEN 100, EPOCHS 45
# Runtime : 192.99790263175964
# Accuracy:  0.75

# CLIENTS 10, HIDDEN 100, EPOCHS 50
# Runtime : 212.79442358016968
# Accuracy:  0.73





