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

CLIENTS = 2
HIDDEN = 2
EPOCHS = 1
print(f'CLIENTS {CLIENTS}, HIDDEN {HIDDEN}, EPOCHS {EPOCHS}')

torch.multiprocessing.set_sharing_strategy('file_system')

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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)
        self.serv = False

    def forward1(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def forward2(self, x):
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x


    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)
        # return F.log_softmax(self.fc4(x), dim=1)  
        # print(self.serv)

        if x.shape == torch.Size([10, 784]):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            self.serv = True
        else:
            # print('hi')
            x = F.relu(self.fc3(x))
            x = F.log_softmax(self.fc4(x), dim=1) 
            self.serv = False
        return x 


models = []
for i in range(CLIENTS):
    m = Net()
    models.append(m)

GlobalNet = Net()

loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(models[0].parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)


start = time.time()
for epoch in range(EPOCHS): 
    i=0
    for data in trainset:  
        X, y = data
        client_outs = []
        for m in range(len(models)):
            client_outs.append(models[m](X.view(-1,784)))
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


        if i%100 == 99:
            print(f'epoch={epoch}, batch={i}')
        i+=1 

print(f'Runtime : {time.time()-start}')

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

print("Accuracy: ", round(correct/total, 2))

PATH = "models/mnist_t.pth"

state = {
    'epoch': 1,
    'state_dict': GlobalNet.state_dict(),
    'optimizer': optimizer.state_dict(),
}
torch.save(state, PATH)










