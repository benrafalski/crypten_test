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


modelA = ClientA()
modelB = ClientB()


model = GlobalModel(modelA, modelB)
# x1, x2 = torch.randn(1, 784), torch.randn(1, 784)
# output = model(x1, x2)

loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)



start = time.time()
for epoch in range(1): 
    i=0
    for dataA, dataB in zip(trainA, trainB):  
        XA, yA = dataA  
        XB, yB = dataB
        output = model(XA.view(-1,784), XB.view(-1,784))  
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


PATH = "models/aggregate_pt.pth"

state = {
    'epoch': 1,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}
torch.save(state, PATH)
