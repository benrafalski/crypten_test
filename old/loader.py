import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from torch.utils.data import Subset
import torch.multiprocessing as mp
import crypten
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, datasets
import torchvision
import torch
import itertools
import time
import torch
import torchvision
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True, num_workers=2)

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False, num_workers=2)


trainset_shape = trainset.dataset.train_data.shape
testset_shape = testset.dataset.test_data.shape

print(trainset_shape, testset_shape)

dataset = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

def make_dataset(size):
    indices = np.arange(len(dataset))
    train_indices, test_indices = train_test_split(indices, train_size=size*10, stratify=dataset.targets)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=2, batch_size=10)
    test_loader = DataLoader(test_dataset, shuffle=False, num_workers=2, batch_size=10)

    return train_loader, test_loader

trainA, testA = make_dataset(3000)
trainB, testB = make_dataset(3000)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)   

class CNet(crypten.nn.Module):
    def __init__(self):
        super(CNet, self).__init__()
        self.layer1 = crypten.nn.Sequential(
            crypten.nn.Linear(28*28, 64),
            crypten.nn.ReLU()
        )
        self.layer2 = crypten.nn.Sequential(
            crypten.nn.Linear(64, 64),
            crypten.nn.ReLU()
        )
        self.layer3 = crypten.nn.Sequential(
            crypten.nn.Linear(64, 64),
            crypten.nn.ReLU()
        )
        self.layer4 = crypten.nn.Sequential(
            crypten.nn.Linear(64, 10),
            crypten.nn.LogSoftmax(dim=1)
        )
        self.serv = False

    def forward(self, x):
        if self.serv == False:
            x = self.layer1(x)
            x = self.layer2(x)
            self.serv = True
        else:
            x = self.layer3(x)
            x = self.layer4(x)
            self.serv = False
        return x

class ClientA(nn.Module):
    def __init__(self):
        super(ClientA, self).__init__()
        self.fc1 = nn.Linear(28*28, 32)
        self.fc2 = nn.Linear(32, 32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class ClientB(nn.Module):
    def __init__(self):
        super(ClientB, self).__init__()
        self.fc1 = nn.Linear(28*28, 32)
        self.fc2 = nn.Linear(32, 32)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class GlobalModel(nn.Module):
    def __init__(self, modelA, modelB):
        super(GlobalModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class Client(crypten.nn.Module):
    def __init__(self):
        super(Client, self).__init__()

        self.layer1 = crypten.nn.Sequential(
            crypten.nn.Linear(28*28, 32),
            crypten.nn.ReLU()
        )
        self.layer2 = crypten.nn.Sequential(
            crypten.nn.Linear(32, 32),
            crypten.nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Global(crypten.nn.Module):
    def __init__(self, modelA, modelB):
        super(Global, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.layer3 = crypten.nn.Sequential(
            crypten.nn.Linear(64, 64),
            crypten.nn.ReLU()
        )
        self.layer4 = crypten.nn.Sequential(
            crypten.nn.Linear(64, 10),
            crypten.nn.LogSoftmax(dim=1)
        )

    def forward(self, x1, x2):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        if(self.encrypted):
            x = crypten.cat([x1, x2], dim=1)
        else:
            x = torch.cat((x1, x2), dim=1)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
# net = Net()
# print(net)


# loss_criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)


# start = time.time()
# for epoch in range(1): 
#     i=0
#     for data in trainset:  
#         X, y = data  
#         output = net(X.view(-1,784))  
#         loss = loss_criterion(output, y)  
#         net.zero_grad() 
#         loss.backward()  
#         optimizer.step() 
#         if i%100 == 99:
#             print(f'epoch={epoch}, batch={i}')
#         i+=1 

# print(f'Runtime : {time.time()-start}')




# print("Accuracy: ", round(correct/total, 2))



# for no mpc or fl the size is 468KB
def load_benchmark():
    PATH = "models/mnist_t.pth"
    state = torch.load(PATH)
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)
    net.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])


    correct = 0
    total = 0

    with torch.no_grad():
        for data in testset:
            X, y = data
            output = net(X.view(-1,784))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 2))

def load_mpc_fl_vertical():
    PATH = "models/mnist_ct.pth"
    state = torch.load(PATH)
    net = CNet()
    optimizer = crypten.optim.SGD(
        net.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)
    net.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])


    correct = 0
    total = 0

    with torch.no_grad():
        net.decrypt()
        for data in testset:
            X, y = data
            client_output = net(X.view(-1, 784))
            output = net(client_output)
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 2))

def load_addition_benchmark():
    PATH = "models/aggregate_pt.pth"
    state = torch.load(PATH)
    modelA = ClientA()
    modelB = ClientB()
    model = GlobalModel(modelA, modelB)
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])


    correct = 0
    total = 0

    with torch.no_grad():
        for dataA, dataB in zip(trainA, trainB):
            XA, yA = dataA  
            XB, yB = dataB
            output = model(XA.view(-1,784), XB.view(-1,784)) 
            for idx, i in enumerate(output):
                if torch.argmax(i) == yA[idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 2))


def load_mpc_fl_additive():
    PATH = "models/aggregate_ct.pth"
    state = torch.load(PATH)
    modelA = Client()
    modelB = Client()
    model = Global(modelA, modelB)
    optimizer = crypten.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])


    correct = 0
    total = 0

    with torch.no_grad():
        modelA.decrypt()
        modelB.decrypt()
        model.decrypt()
        for dataA, dataB in zip(testA, testB):
            XA, yA = dataA
            XB, yB = dataB
            output = model(XA.view(-1,784), XB.view(-1,784)) 
            for idx, i in enumerate(output):
                if torch.argmax(i) == yA[idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 2))

load_mpc_fl_additive()












