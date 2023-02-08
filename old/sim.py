import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import crypten 
import time

crypten.init()

n_input, n_hidden, n_out, batch_size, learning_rate = 10, 15, 1, 5, 0.01

print(f'\n\n input layer size: {n_input}\n hidden layer size: {n_hidden}\n output layer size: {n_out}\n batch_size: {batch_size}\n learning rate: {learning_rate}')

data_x = torch.randn(batch_size, n_input)
data_y = (torch.rand(size=(batch_size, 1)) < 0.5).float()
print(f'types: {type(data_x)}, {type(data_y)}')

data_x_enc = crypten.cryptensor(data_x)
data_y_enc = crypten.cryptensor(data_y)


print(f"\n\n Is the training data encrypted? {crypten.is_encrypted_tensor(data_x_enc)}")
print(f" Is the testing data encrypted? {crypten.is_encrypted_tensor(data_y_enc)}")


class NN(crypten.nn.Module): 
    def __init__(self): 
        super().__init__()
        self.layer1 = crypten.nn.Sequential(
            crypten.nn.Linear(n_input, n_hidden),
            crypten.nn.ReLU()
            # crypten.nn.Sigmoid()
        )
        self.layer2 = crypten.nn.Sequential(
            crypten.nn.Linear(n_hidden, n_out),
            crypten.nn.ReLU()
        )
        self.serv = 1
        
    def forward(self, x):
        if x.grad_fn == None:
            x = self.layer1(x)
            if self.serv == 2:
                x = self.layer2(x)
            self.serv = 2
        else: 
            x = self.layer2(x)
        return x


model = NN()
model.encrypt()
print(f" The client-side model is: {model}\n\n")

print(f' Training the client model on the first layer only...')
pred_y = model(data_x_enc)


print(f' Sending the model to the server...')
server_model = model

print(f' Finishing second layer training on the server side...')
loss_function = crypten.nn.MSELoss()
optimizer = crypten.optim.SGD(server_model.parameters(), lr=learning_rate)
pred_y = server_model(data_x_enc)
loss = loss_function(pred_y, data_y_enc)
model.zero_grad()
loss.backward()
print(f'\n Loss function results on the server side after second layer: {loss.get_plain_text()}')
model.update_parameters(learning_rate)


print(f'\n Continuing training for 500 epochs on server side...')
losses = []
start_time = time.time()
for epoch in range(500):
    pred_y = server_model(data_x_enc)
    loss = loss_function(pred_y, data_y_enc)
    losses.append(loss.get_plain_text().item())
    model.zero_grad()
    loss.backward()

    if epoch % 100 == 99:
            print(f'\tepoch: {epoch}, loss: {loss.get_plain_text()}')

    model.update_parameters(learning_rate)

print(f"\n Training time: {time.time()-start_time} seconds")



