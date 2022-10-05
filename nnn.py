import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import crypten 

crypten.init()

# MNIST 
digits = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
digits_test = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)

# precprocess into tensors
def take_samples(digits, n_samples=1000):
    images, labels = [], [] 
    for i, digit in enumerate(digits): 
        if i == n_samples: 
            image, label = digit 
            images.append(image)
            label_one_hot = F.one_hot(torch.tensor(label), 10)
            labels.append(label_one_hot)

    images = torch.cat(images)
    labels = torch.stack(labels)
    return images, labels
images, labels = take_samples(digits, n_samples=100)
# encrypt data 
images_enc = crypten.cryptensor(images)
labels_enc = crypten.cryptensor(labels)
# print(f'IMAGES {images} ')
# print(f'ENC IMAGES {images_enc}')

# test set
images_test, labels_test = take_samples(digits, n_samples=20)
images_test_enc = crypten.cryptensor(images)
labels__test_enc = crypten.cryptensor(labels)

def train_model(model, X, y, epochs=10, learning_rate=0.05): 
    criterion = crypten.nn.CrossEntropyLoss() 
    for epoch in range(epochs): 
        model.zero_grad() 
        output = model(X) 
        loss = criterion(output, y)
        print(f'epoch {epoch} loss: {loss.get_plain_test()}')
        loss.backward() 
        model.update_parameters(learning_rate)
    return model 

class NN(crypten.nn.Module): 
    def __init__(self): 
        super().__init__()
        self.conv1 = crypten.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = crypten.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = crypten.nn.Dropout2d(0.25)
        self.dropout2 = crypten.nn.Dropout2d(0.5)
        self.fc1 = crypten.nn.Linear(9216, 128)
        self.fc2 = crypten.nn.Linear(128, 10)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)
        x = x.relu()
        x = x.max_pool2d(2)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = x.relu()
        x = self.dropout2(x)
        x = self.fc2(x)
        return x 

model = NN().encrypt()
x = images_enc[0].unsqueeze(0)
print(x.shape)
model(x)

# model = train_model(model, images_enc[:10, ], labels_enc[:10,], epochs=3)



        











