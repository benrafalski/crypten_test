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
from multiprocessing import Pool
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

# clients -> 1000 use 85 for 0.656 accuracy
# clients -> 500 use 100 for 0.751 accuracy
# clients -> 100 use 120 for 0.862 accuracy
# cleints -> 10 use 20 for 0.951 accuracy

CLIENTS = 2
SIZE = 6000//CLIENTS
EPOCHS = 5
print(f'CLIENTS {CLIENTS}, SIZE {SIZE}, EPOCHS {EPOCHS}')

torch.multiprocessing.set_sharing_strategy('file_system')

# train = datasets.MNIST('', train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor()
#                        ]))

# test = datasets.MNIST('', train=False, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor()
#                        ]))

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
# print(f'train = {trainset_shape}\n test = {testset_shape}')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)
        self.serv = False

    def forward(self, x):
        if x.shape == torch.Size([10, 784]):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            self.serv = True
        else:
            # print('hi')
            x = F.relu(self.fc3(x))
            x = F.log_softmax(self.fc4(x), dim=1) 
            # x = torch.sigmoid(self.fc4(x))
            self.serv = False
        return x 


models = []
for i in range(CLIENTS):
    m = Net()
    models.append(m)

GlobalNet = Net()

loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(models[0].parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)

def train_client(args):
    X, m = args
    mod = models[m](X.view(-1,784))
    return mod

start = time.time()
client_train = 0
total_client_time = 0

server_time = 0
total_server_time = 0

for epoch in range(EPOCHS): 
    i=0
    e_time = 0
    se_time = time.time()
    for data in trainset:  
        X, y = data
        client_outs = []
        index = list(range(len(models)))
        args = zip(repeat(X), index)

        for m in range(len(models)):
            if m == 0: 
                cstart = time.time()
                mod = train_client((X, m))
                client_train = time.time() - cstart
            else:
                mod = train_client((X, m))
            client_outs.append(mod)

        total_client_time = total_client_time + client_train

# start training -> first client finishes - client ends -> end training

        server_time = time.time()

        # transfer network to server
        GlobalNet = models[0]
        # finish last 2 layers in server side
        output = GlobalNet(client_outs[0])
        # send network back to client side to update parameters and repeat
        models[0] = GlobalNet

        loss = loss_criterion(output, y)  
        GlobalNet.zero_grad() 
        loss.backward()  
        optimizer.step() 


        # if i%100 == 99:
        #     print(f'epoch={epoch}, batch={i}')
        i+=1 
        total_server_time = total_server_time + (time.time()- server_time)
        
    print(f'EPOCH {epoch} runtime : {time.time()-se_time}')


print(f'client time = {total_client_time}, server time = {total_server_time}')
print(f'Runtime : {total_server_time+total_client_time}')

correct = 0
total = 0

with torch.no_grad():
    for data in testset:  
        X, y = data
        client_outs = []
        for m in range(len(models)):
            client_outs.append(models[m](X.view(-1,784)))
        output = models[0](client_outs[0])
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))

PATH = "models/mnist_t.pth"

state = {
    'epoch': 1,
    'state_dict': GlobalNet.state_dict(),
    'optimizer': optimizer.state_dict(),
}
torch.save(state, PATH)










