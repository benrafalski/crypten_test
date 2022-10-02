import crypten
import torch
import torch.nn as nn
import torch.nn.functional as F

crypten.init()
torch.set_num_threads(1)



#Define an example network
class ExampleNet(nn.Module):
    pass_num = 1
    def __init__(self, pass_num):
        super(ExampleNet, self).__init__()
        self.pass_num = pass_num
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(16 * 12 * 12, 100)
        self.fc2 = nn.Linear(100, 2) # For binary classification, final layer needs only 2 outputs
 

    def inc_pass(self):
        self.pass_num += 1
        print(f'new pas num : {self.pass_num}')
    
    def forward(self, x):  
        print(f'calling forward() func')
        if self.pass_num == 1: 
            out = self.conv1(x)
            out = F.relu(out)
            return out
        elif self.pass_num == 2:
            out = F.max_pool2d(x, 2)
            out = out.view(-1, 16 * 12 * 12)
            out = self.fc1(out)
            out = F.relu(out)
            out = self.fc2(out)
            return out
        else:
            return self.forwardserver(x)

    def forward1(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        self.pass_num = 2
        return out

    def forward2(self, x): 
        out = F.max_pool2d(x, 2)
        out = out.view(-1, 16 * 12 * 12)
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        self.pass_num = 3
        return out

    def forwardserver(self, x): 
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 16 * 12 * 12)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
    
crypten.common.serial.register_safe_class(ExampleNet)

# Define source argument values for Alice and Bob
ALICE = 0
BOB = 1

data_alice_enc = crypten.load_from_party('/tmp/alice_train.pth', src=ALICE)

# We'll now set up the data for our small example below
# For illustration purposes, we will create toy data
# and encrypt all of it from source ALICE
x_small = torch.rand(100, 1, 28, 28)
y_small = torch.randint(1, (100,))

# Transform labels into one-hot encoding
label_eye = torch.eye(2)
y_one_hot = label_eye[y_small]

# Transform all data to CrypTensors
x_train = crypten.cryptensor(x_small, src=ALICE)
y_train = crypten.cryptensor(y_one_hot)

# Instantiate and encrypt a CrypTen model
model_plaintext = ExampleNet(1)
dummy_input = torch.empty(1, 1, 28, 28)
model = crypten.nn.from_pytorch(model_plaintext, dummy_input)
model.encrypt()


model.train() # Change to training mode
loss = crypten.nn.MSELoss() # Choose loss functions

# Set parameters: learning rate, num_epochs
learning_rate = 0.001
num_epochs = 2

# Train the model: SGD on encrypted data

# forward pass client
output_client = model(x_train)
print(f'outclient: {output_client}')
# output_firstpass = model(output_client)
# loss_value = loss(output_firstpass, y_train)
    

model_plaintext = ExampleNet(2)
dummy_input = torch.empty(1, 1, 28, 28)
model = crypten.nn.from_pytorch(model_plaintext, dummy_input)
model.encrypt()


model.train() # Change to training mode
loss = crypten.nn.MSELoss() # Choose loss functions

# Set parameters: learning rate, num_epochs
learning_rate = 0.001
num_epochs = 2
out = model(output_client)

# # set gradients to zero
# model.zero_grad()

# # perform backward pass
# loss_value.backward()

# # update parameters
# model.update_parameters(learning_rate) 

# # examine the loss after each epoch
# print(f"Epoch: 'client' Loss: {loss_value.get_plain_text()}")

# for i in range(num_epochs):

#     # forward pass
#     output = model(x_train)
#     loss_value = loss(output, y_train)
    
#     # set gradients to zero
#     model.zero_grad()

#     # perform backward pass
#     loss_value.backward()

#     # update parameters
#     model.update_parameters(learning_rate) 
    
#     # examine the loss after each epoch
#     print("Epoch: {0:d} Loss: {1:.4f}".format(i, loss_value.get_plain_text()))











