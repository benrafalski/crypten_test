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

crypten.init()

n_input, n_hidden, n_out, batch_size, learning_rate = 4, 25, 3, 5, 0.001

# data stuff?
iris = load_iris()
print(type(iris))
data = iris['data']
labels = iris['target']
label_names = iris['target_names']
feature_names = iris['feature_names']

scaler = StandardScaler()
data = scaler.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=4)

# print(X_train)

X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

# targets_one_hot = torch.nn.functional.one_hot(targets, num_classes)
# T = crypten.cryptensor(targets_one_hot)

y_train_one_hot = torch.nn.functional.one_hot(y_train_tensor)
y_test_one_hot = torch.nn.functional.one_hot(y_test_tensor)


X_train_enc = crypten.cryptensor(X_train_tensor)
X_test_enc = crypten.cryptensor(X_test_tensor)
y_train_enc = crypten.cryptensor(y_train_one_hot)
y_test_enc = crypten.cryptensor(y_test_one_hot)


# X_train = Variable(torch.from_numpy(X_train)).float()


# print(f'xtrain={X_train}')
# print(f'xtest={X_test}')
# print(f'ytrain={y_train}')
# print(f'ytest={y_test}')


# training_data = data.sample(frac=0.8, random_state=25)
# testing_data = data.drop(training_data.index)
# train_tensor = torch.tensor(training_data.values)
# test_tensor = torch.tensor(testing_data.values)


# train_enc=0
# test_enc=0
# train_enc = crypten.cryptensor(train_tensor)
# test_enc = crypten.cryptensor(test_tensor)


print(
    f"\n\n Is the training data encrypted? {crypten.is_encrypted_tensor(X_train_enc)}")
print(
    f" Is the testing data encrypted? {crypten.is_encrypted_tensor(X_test_enc)}")


class NN(crypten.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = crypten.nn.Sequential(
            crypten.nn.Linear(n_input, n_hidden),
            crypten.nn.ReLU()
        )
        self.layer2 = crypten.nn.Sequential(
            crypten.nn.Linear(n_hidden, n_hidden),
            crypten.nn.ReLU()
        )
        self.layer3 = crypten.nn.Sequential(
            crypten.nn.Linear(n_hidden, n_out),
            crypten.nn.Softmax(1)
            # crypten.nn.Sigmoid()
        )
        self.serv = 1

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


model = NN()
model.encrypt()
model.train()

loss_function = crypten.nn.CrossEntropyLoss()
optimizer = crypten.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-6)

EPOCHS = 10
losses = []
start_time = time.time()
for epoch in range(EPOCHS):
    pred_y = model(X_train_enc)
    loss = loss_function(pred_y, y_train_enc)
    losses.append(loss.get_plain_text().item())
    # optimizer.zero_grad()
    model.zero_grad()
    loss.backward()

    if epoch % 10 == 9:
        print(f'\tepoch: {epoch}, loss: {loss.get_plain_text()}')

    # server_model.update_parameters(learning_rate)
    optimizer.step()

    with torch.no_grad():
        y_pred = model(X_test_enc)
        correct = (torch.argmax(y_pred.get_plain_text(), dim=1) == y_test_tensor).type(torch.FloatTensor)
        # print(correct.mean())

print(f"\n Training time: {time.time()-start_time} seconds")

test_loss = 0
correct = 0

output = model(X_test_enc)
with torch.no_grad():
    _, predictions = output.max(0)
    prediction_tensor = torch.argmax(predictions.get_plain_text(), dim=1)
    for i in range(30): 
        print(f'Expected: {label_names[y_test_tensor[i].item()]} vs. real: {label_names[prediction_tensor[i].item()]}')
    correct = (prediction_tensor == y_test_tensor).sum()
    samples = predictions.size(0)
    print(f'accuracy: {correct}/{samples} {float(correct)/float(samples) * 100:.2f}%')




