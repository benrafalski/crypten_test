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

train = []
test = []

for i in range(8):
    a, b = make_dataset(750)
    train.append(a)
    test.append(b)




# class ClientA(nn.Module):
#     def __init__(self):
#         super(ClientA, self).__init__()
#         self.fc1 = nn.Linear(28*28, 32)
#         self.fc2 = nn.Linear(32, 32)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return x


class Client(nn.Module):
    def __init__(self, n):
        super(Client, self).__init__()
        self.fc1 = nn.Linear(28*28, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class GlobalModel(nn.Module):
    def __init__(self, models):
        super(GlobalModel, self).__init__()
        self.models = models
        # self.modelB = modelB
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, c):
        # x1 = self.modelA(x1)
        # x2 = self.modelB(x2)

        for i in range(len(c)):
            c[i] = self.models[i](c[i])
        

        x = torch.cat(c, dim=1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


# modelA = ClientA()
# modelB = ClientB()


models = []
for i in range(8):
    m = Client(8)
    models.append(m)


model = GlobalModel(models)
# x = []
# for i in range(8):
#     x.append(torch.randn(1, 784))
# # x1, x2 = torch.randn(1, 784), torch.randn(1, 784)
# output = model(x)
# print(output)


loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)

# zipped = zip(train)


# print(f'zipped = {list(zipped)}')
# z = zip(train[0], train[1])
# print(f'old way = {list(zip(trainA, trainB))}')


# def append_data(data, X, y):
#     a, b = data0
#     X.append(a)
#     y.append(b)
#     return X, y

# for data1, data2 in zip(trainA, trainB):
#     XA, ya = data1
#     print(f'x = {XA.shape}')

# for epoch in range(1): 
#     i=0
#     for data1, data2 in zip(trainA, trainB):
#         XA, ya = data1
        

#         output = model(X)  
#         loss = loss_criterion(output, y[0])  
#         model.zero_grad() 
#         loss.backward()  
#         optimizer.step() 
#         if i%100 == 99:
#             print(f'epoch={epoch}, batch={i}')
#         i+=1 


start = time.time()
for epoch in range(1): 
    batch=0
    for data0, data1, data2, data3, data4, data5, data6, data7 in zip(train[0], train[1], train[2], train[3], train[4], train[5], train[6], train[7]):  
        data = []
        data.append(data0)
        data.append(data1)
        data.append(data2)
        data.append(data3)
        data.append(data4)
        data.append(data5)
        data.append(data6)
        data.append(data7)

        X = []
        y = []    
        for i in range(8):
            a, b = data[i]
            # print(a.shape)
            X.append(a.view(-1, 784))
            y.append(b)

        # print(X)

        output = model(X)  
        loss = loss_criterion(output, y[0])  
        model.zero_grad() 
        loss.backward()  
        optimizer.step() 
        if batch%100 == 99:
            print(f'epoch={epoch}, batch={batch}')
        batch+=1 

    
print(f'Runtime : {time.time()-start}')


correct = 0
total = 0

with torch.no_grad():
    for data0, data1, data2, data3, data4, data5, data6, data7 in zip(test[0], test[1], test[2], test[3], test[4], test[5], test[6], test[7]):  
        data = []
        data.append(data0)
        data.append(data1)
        data.append(data2)
        data.append(data3)
        data.append(data4)
        data.append(data5)
        data.append(data6)
        data.append(data7)

        X = []
        y = []    
        for i in range(8):
            a, b = data[i]
            # print(a.shape)
            X.append(a.view(-1, 784))
            y.append(b)

        
        output = model(X) 
        for idx, i in enumerate(output):
            for why in y:
                if torch.argmax(i) == why[idx]:
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
