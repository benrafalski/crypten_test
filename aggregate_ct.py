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

    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=2, batch_size=10)
    test_loader = DataLoader(test_dataset, shuffle=False, num_workers=2, batch_size=10)

    return train_loader, test_loader

trainA, testA = make_dataset(3000)
trainB, testB = make_dataset(3000)








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


class GlobalModel(crypten.nn.Module):
    def __init__(self, modelA, modelB):
        super(GlobalModel, self).__init__()
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


modelA = Client()
modelB = Client()

modelA.encrypt()
modelB.encrypt()


model = GlobalModel(modelA, modelB)
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
for epoch in range(1): 
    i=0
    for dataA, dataB in zip(trainA, trainB):  
        XA, yA = enc_data(dataA)
        XB, yB = enc_data(dataB)
        output = model(XA.view(-1,784), XB.view(-1,784))  
        if(output.size() != yA._tensor.size()):
            continue
        loss = loss_criterion(output, yA)  
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
# correct = 0
# total = 0
# for dataA, dataB in zip(testA, testB):
#     XA, yA = enc_data(dataA)
#     XB, yB = enc_data(dataB)
#     output = model(XA.view(-1,784), XB.view(-1,784))
#     for idx, i in enumerate(output.get_plain_text()):
        
#         print(output.get_plain_text().argmax(1))
#         print(yA.get_plain_text()[idx])
#         if torch.argmax(i) == yA.get_plain_text()[idx]:
#             correct += 1
#         total += 1



