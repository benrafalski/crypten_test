import crypten 
import torch 
import torch.nn as nn
# import crypten.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms

crypten.init()

training_data = pd.read_csv("")








# get data 
# train_data = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
# test_data = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

# preprocess data
# images, labels = [], []
# for i, data in enumerate(train_data): 
#     if i == 100:
#         break 
#     image, label = data 
#     images.append(image)
#     label_one_hot = torch.nn.functional.one_hot(torch.tensor(label), 10)
#     labels.append(label_one_hot)

# images = torch.cat(images) # ([100, 28, 28])
# labels = torch.stack(labels) # ([100, 10])

# images_enc = crypten.cryptensor(images)
# labels_enc = crypten.cryptensor(labels)

class NNet(crypten.nn.Module):
    def __init__(self, input_size, num_classes):
        super(NNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self, x):
        # if x.grad_fn == None:
        #     out = F.relu(self.fc1(x))
        # else:
        #     out = self.fc2(x)
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out



# run 64 images from mnist at the same time each batch
# x = torch.randn(64, 784)
# x_enc = crypten.cryptensor(x)
# # # print(public_model(x).shape) # ([64, 10])

# criterion = crypten.nn.CrossEntropyLoss()
# for epoch in range(1): 
#     private_model.zero_grad() 
#     output = private_model(images_enc)
#     loss = criterion(output, labels_enc)
#     print(f'epoch {epoch} loss: {loss.get_plain_text()}')
#     loss.backward()
#     private_model.update_parameters(0.05)

# hyper params 
input_size = 784
num_classes = 10 
learning_rate = 0.001
batch_size = 64 
num_epochs = 1

# load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# init network
# model = NNet(input_size=input_size, num_classes=num_classes)


criterion = nn.CrossEntropyLoss()
# optimizer = crypten.optim(private_model.parameters(), lr=learning_rate)


# client side 
model = NNet(784, 10)
private_model = model.encrypt()

print(train_loader[0])




# trian network 
for epoch in range(num_epochs): 
    print(f'epoch: {epoch}')
    for batch_idx, (data, targets) in enumerate(train_loader): 
        # reshape
        data = data.reshape(data.shape[0], -1)
        # forward
        scores = private_model(data)
        # loss = criterion(scores, torch.nn.functional.one_hot(torch.tensor(targets), 10))
        loss = criterion(scores, targets)
        if batch_idx % 100 == 99:
            print(f'\tbatch: {batch_idx}, loss: {loss}')
        # backward
        private_model.zero_grad() 
        # private_model.zero_grad()
        loss.backward() 
        # gradient descent or adam step 
        # optimizer.step() 


# crypten.init()

# class Net(nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 5x5 square convolution
#         # kernel
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square, you can specify with a single number
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# net = Net()
# print(net)
# dummy_input = torch.empty(1, 1, 28, 28)
# model = crypten.nn.from_pytorch(model_plaintext, dummy_input)
# model.encrypt()
# net_enc = net.encrypt()

# # params = list(net.parameters())
# # print(len(params))
# # print(params[0].size())  # conv1's .weight

# # input = torch.randn(1, 1, 32, 32)
# # out = net(input)
# # print(out)

# # net.zero_grad()
# # out.backward(torch.randn(1, 10))











