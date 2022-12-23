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
from sklearn.model_selection import train_test_split
import numpy as np
from statistics import mean

CLIENTS = 500
SIZE = 6000//CLIENTS
EPOCHS = 100
print(f'CLIENTS {CLIENTS}, SIZE {SIZE}, EPOCHS {EPOCHS}')

crypten.init()

dataset = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

indices = np.arange(len(dataset))
train_indices, test_indices = train_test_split(indices, train_size=SIZE*10, test_size=10000, stratify=dataset.targets)
train = Subset(dataset, train_indices)
test = Subset(dataset, test_indices)


trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True, num_workers=2)

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False, num_workers=2)


# trainset_shape = trainset.dataset.train_data.shape
# testset_shape = testset.dataset.test_data.shape

# print(trainset_shape, testset_shape)


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
        if x.shape == torch.Size([10, 784]):
            x = self.layer1(x)
            x = self.layer2(x)
            self.serv = True
        else:
            
            x = self.layer3(x)
            x = self.layer4(x)


        self.serv = not self.serv
        return x


# net = Net()
# net.encrypt()
# # print(net)
# net.train()

client_net = []
for i in range(CLIENTS):
    model = Net()
    model.encrypt()
    model.train()
    client_net.append(model)



loss_criterion = crypten.nn.CrossEntropyLoss()
optimizer = crypten.optim.SGD(
    client_net[0].parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)



def train():
    client_runtimes = []
    server_runtimes = []
    epoch_runtimes = []
    for epoch in range(EPOCHS):
        print(f'Starting epoch #{epoch+1}')
        epoch_start = time.time()
        for data in trainset:
            # encrypt the data
            X1, y1 = data
            x1_enc = crypten.cryptensor(X1.view(-1, 784))
            y1_one_hot = torch.nn.functional.one_hot(y1)
            y1_enc = crypten.cryptensor(y1_one_hot)

            # train first 2 layers using client
            
            client_output = []
            for h in range(CLIENTS):
                if h == 0:
                    client_start = time.time()
                    c = client_net[h](x1_enc)
                    client_end = time.time() - client_start
                    client_runtimes.append(client_end)
                else:
                    c = client_net[h](x1_enc)

                client_output.append(c)


            server_start = time.time()
            # transfer network to server
            net = client_net[0]
            # finish last 2 layers in server side
            output = net(client_output[0])
            # send network back to client side to update parameters and repeat
            client_net[0] = net


            if(output.size() != y1_enc._tensor.size()):
                continue
            loss = loss_criterion(output, y1_enc)
            net.zero_grad()
            loss.backward()
            optimizer.step()
            server_end = time.time() - server_start
            server_runtimes.append(server_end)

        epoch_end = time.time() - epoch_start
        epoch_runtimes.append(epoch_end)

    total_runtime_all_epochs = sum(epoch_runtimes)
    avg_epoch_runtime = mean(epoch_runtimes)
    avg_server_runtime = mean(server_runtimes)
    avg_client_runtime = mean(client_runtimes)
    avg_runtime = avg_server_runtime + avg_client_runtime
    return total_runtime_all_epochs, avg_epoch_runtime, avg_server_runtime, avg_client_runtime, avg_runtime




start = time.time()

total_runtime_all_epochs, avg_epoch_runtime, avg_server_runtime, avg_client_runtime, avg_runtime = train()

print(f"total_runtime_all_epochs : {total_runtime_all_epochs}")
print(f"avg_epoch_runtime : {avg_epoch_runtime}")
print(f"avg_server_runtime : {avg_server_runtime}")
print(f"avg_client_runtime : {avg_client_runtime}")  
print(f"avg_runtime : {avg_runtime}")


# testing model accuracy
correct = 0
total = 0
client_net[0].eval()

with torch.no_grad():
    client_net[0].decrypt()
    for data in testset:
        X, y = data
        client_output = client_net[0](X.view(-1, 784))
        output = client_net[0](client_output)
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 2))

PATH = "models/mnist_ct.pth"

state = {
    'epoch': 1,
    'state_dict': client_net[0].state_dict(),
    'optimizer': optimizer.state_dict(),
}
torch.save(state, PATH)
# Clients   Runtime   Accuracy  Epochs
# 2         ?         0.??      1
# 10        1769.31   0.93      1
# 100       9293.01   0.92      1
# 500       520.25    0.10      1
# 1000      520.40    0.10      1 





# clients -> 1000 use 85 for 0.656 accuracy

# clients -> 500 use 100 for 0.751 accuracy

# clients -> 100 use 120 for 0.862 accuracy
    # total_runtime_all_epochs : 11614.416815519333
    # avg_epoch_runtime : 96.78680679599444
    # avg_server_runtime : 0.2015748949540486
    # avg_client_runtime : 0.014472565518485175
    # avg_runtime : 0.21604746047253376
# cleints -> 10 use 20 for 0.951 accuracy
    # total_runtime_all_epochs : 3367.824250936508
    # avg_epoch_runtime : 168.39121254682541
    # avg_server_runtime : 0.19396333562768875
    # avg_client_runtime : 0.013238914926846822
    # avg_runtime : 0.20720225055453556