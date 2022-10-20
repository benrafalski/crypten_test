import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import crypten
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import time
import torch


transforms = transforms.Compose([transforms.ToTensor()])

n_input, n_hidden, n_out, batch_size, learning_rate = 4, 5, 3, 5, 0.001

iris = load_iris()
print(type(iris))
data = iris['data']
labels = iris['target']
label_names = iris['target_names']
feature_names = iris['feature_names']

# scaler = StandardScaler()
# data = scaler.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=2)

X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

# y_train_one_hot = torch.nn.functional.one_hot(y_train_tensor)
# y_test_one_hot = torch.nn.functional.one_hot(y_test_tensor)


# X_train_enc = crypten.cryptensor(X_train_tensor)
# X_test_enc = crypten.cryptensor(X_test_tensor)
# y_train_enc = crypten.cryptensor(y_train_one_hot)
# y_test_enc = crypten.cryptensor(y_test_one_hot)


# print(
#     f"\n\n Is the training data encrypted? {crypten.is_encrypted_tensor(X_train_enc)}")
# print(
#     f" Is the testing data encrypted? {crypten.is_encrypted_tensor(X_test_enc)}")


class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden),
            torch.nn.Sigmoid()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, n_out),
            torch.nn.Sigmoid()
        )
        self.serv = 1

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


model = NN()
model.train()

pred = model(X_train_tensor)

server_model = model

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(server_model.parameters(), lr=learning_rate)
pred = server_model(X_train_tensor)
loss = loss_function(pred, y_train_tensor)
server_model.zero_grad()
loss.backward()
print(
    f'\n Loss function results on the server side after second layer: {loss}')
server_model.update_parameters(learning_rate)

EPOCHS = 5
print(f'\n Continuing training for {EPOCHS} epochs on server side...')
losses = []
start_time = time.time()
for epoch in range(EPOCHS):
    pred_y = server_model(X_train_tensor)
    loss = loss_function(pred_y, y_train_tensor)
    losses.append(loss.item())
    server_model.zero_grad()
    loss.backward()

    if epoch % 100 == 99:
        print(f'\tepoch: {epoch}, loss: {loss}')

    server_model.update_parameters(learning_rate)

print(f"\n Training time: {time.time()-start_time} seconds")



# def test(model, device, test_loader):
# server_model.eval()
# test_loss = 0
# correct = 0

# output = model(X_test_enc)
# with torch.no_grad():
#     _, predictions = output.max(0)
#     prediction_tensor = torch.argmax(predictions.get_plain_text(), dim=1)
#     for i in range(30): 
#         print(f'Expected: {label_names[y_test_tensor[i].item()]} vs. real: {label_names[prediction_tensor[i].item()]}')
#     correct = (prediction_tensor == y_test_tensor).sum()
#     samples = predictions.size(0)
#     print(f'accuracy: {correct}/{samples} {float(correct)/float(samples) * 100:.2f}%')



