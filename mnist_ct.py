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



crypten.init()

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


class Net(crypten.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


# net = Net()
# net.encrypt()
# # print(net)
# net.train()
client_net = Net()
client_net.encrypt()
client_net.train()

loss_criterion = crypten.nn.CrossEntropyLoss()
optimizer = crypten.optim.SGD(
    client_net.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)


def train(client_net):
    i = 0
    for data in trainset:
        # encrypt the data
        X1, y1 = data
        x1_enc = crypten.cryptensor(X1.view(-1, 784))
        y1_one_hot = torch.nn.functional.one_hot(y1)
        y1_enc = crypten.cryptensor(y1_one_hot)

        

        # train first 2 layers using client
        client_output = client_net(x1_enc)
        # transfer network to server
        net = client_net
        # finish last 2 layers in server side
        output = net(client_output)
        # send network back to client side to update parameters and repeat
        client_net = net

        if(output.size() != y1_enc._tensor.size()):
            continue
        loss = loss_criterion(output, y1_enc)
        net.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 99:
            print(f'epoch={1}, batch={i}')
        i += 1
        # # stop after 1200*10 samples
        # if i == 1200:
        #     break


start = time.time()
train(client_net)
print(f"Runtime: {time.time() - start}")


# testing model accuracy
correct = 0
total = 0
client_net.eval()

with torch.no_grad():
    client_net.decrypt()
    for data in testset:
        X, y = data
        client_output = client_net(X.view(-1, 784))
        output = client_net(client_output)
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 2))
