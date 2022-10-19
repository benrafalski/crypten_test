import crypten
import torch
import torch.nn as nn
import torch.nn.functional as F
import crypten.mpc as mpc
import crypten.communicator as comm

crypten.init()
torch.set_num_threads(1)

#Define an example network
class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.fc1 = nn.Linear(16 * 12 * 12, 100)
        self.fc2 = nn.Linear(100, 2) # For binary classification, final layer needs only 2 outputs
 
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 16 * 12 * 12)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
    
crypten.common.serial.register_safe_class(ExampleNet)

def get_global():
    model_plaintext = ExampleNet()
    dummy_input = torch.empty(1, 1, 28, 28)
    model = crypten.nn.from_pytorch(model_plaintext, dummy_input)
    model.encrypt() # this is the global model
    model.train() # Change to training mode
    loss = crypten.nn.MSELoss() # Choose loss functions
    return model

# global model
glbl = get_global()

# send data to clients, now each client has the global data
alice_model = glbl
bob_model = glbl

# get data from clients
ALICE = 0 # client 0
BOB = 1 # client 1
data_alice_enc = crypten.load_from_party('/tmp/alice_train.pth', src=ALICE) # data from client 0
data_bob_enc = crypten.load_from_party('/tmp/bob_train.pth', src=BOB)

# train alice//test
x_small = torch.rand(100, 1, 28, 28) # random data
y_small = torch.randint(1, (100,)) # more random data
label_eye = torch.eye(2)
y_one_hot = label_eye[y_small]
# print(y_one_hot)
x_train = crypten.cryptensor(x_small, src=ALICE)
y_train = crypten.cryptensor(y_one_hot)
alice_model.train() # Change to training mode
alice_loss = crypten.nn.MSELoss() # Choose loss functions
learning_rate = 0.001
num_epochs = 2
for i in range(num_epochs):

    # forward pass
    output = alice_model(x_train)
    loss_value = alice_loss(output, y_train)
    
    # set gradients to zero
    alice_model.zero_grad()

    # perform backward pass
    loss_value.backward()

    # update parameters
    alice_model.update_parameters(learning_rate) 
    
    # examine the loss after each epoch
    print("Epoch: {0:d} Loss: {1:.4f}".format(i, loss_value.get_plain_text()))

# train bob
x_small = torch.rand(100, 1, 28, 28) # random data
y_small = torch.randint(1, (100,)) # more random data
label_eye = torch.eye(2)
y_one_hot = label_eye[y_small]
# print(y_one_hot)
x_train = crypten.cryptensor(x_small, src=BOB)
y_train = crypten.cryptensor(y_one_hot)
bob_model.train() # Change to training mode
bob_loss = crypten.nn.MSELoss() # Choose loss functions
learning_rate = 0.001
num_epochs = 2
for i in range(num_epochs):

    # forward pass
    output = bob_model(x_train)
    loss_value = bob_loss(output, y_train)
    
    # set gradients to zero
    bob_model.zero_grad()

    # perform backward pass
    loss_value.backward()

    # update parameters
    bob_model.update_parameters(learning_rate) 
    
    # examine the loss after each epoch
    print("Epoch: {0:d} Loss: {1:.4f}".format(i, loss_value.get_plain_text()))

# aggregate data




# Define source argument values for Alice and Bob
ALICE = 0
BOB = 1

# Load Alice's data 
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
model_plaintext = ExampleNet()
dummy_input = torch.empty(1, 1, 28, 28)
model = crypten.nn.from_pytorch(model_plaintext, dummy_input)
model.encrypt() # this is the global model

# Example: Stochastic Gradient Descent in CrypTen

model.train() # Change to training mode
loss = crypten.nn.MSELoss() # Choose loss functions

# Set parameters: learning rate, num_epochs
learning_rate = 0.001
num_epochs = 2

# Train the model: SGD on encrypted data
for i in range(num_epochs):

    # forward pass
    output = model(x_train)
    loss_value = loss(output, y_train)
    
    # set gradients to zero
    model.zero_grad()

    # perform backward pass
    loss_value.backward()

    # update parameters
    model.update_parameters(learning_rate) 
    
    # examine the loss after each epoch
    print("Epoch: {0:d} Loss: {1:.4f}".format(i, loss_value.get_plain_text()))





# Convert labels to one-hot encoding
# Since labels are public in this use case, we will simply use them from loaded torch tensors
labels = torch.load('/tmp/train_labels.pth')
labels = labels.long()
labels_one_hot = label_eye[labels]

@mpc.run_multiprocess(world_size=2)
def run_encrypted_training():
    # Load data:
    x_alice_enc = crypten.load_from_party('/tmp/alice_train.pth', src=ALICE)
    x_bob_enc = crypten.load_from_party('/tmp/bob_train.pth', src=BOB)
    
    crypten.print(x_alice_enc.size())
    crypten.print(x_bob_enc.size())
    
    # Combine the feature sets: identical to Tutorial 3
    x_combined_enc = crypten.cat([x_alice_enc, x_bob_enc], dim=2)
    
    # Reshape to match the network architecture
    x_combined_enc = x_combined_enc.unsqueeze(1)
    
    
    # Commenting out due to intermittent failure in PyTorch codebase
    
    # Initialize a plaintext model and convert to CrypTen model
    pytorch_model = ExampleNet()
    model = crypten.nn.from_pytorch(pytorch_model, dummy_input)
    model.encrypt()
    # Set train mode
    model.train()
  
    # Define a loss function
    loss = crypten.nn.MSELoss()

    # Define training parameters
    learning_rate = 0.001
    num_epochs = 2
    batch_size = 10
    num_batches = x_combined_enc.size(0) // batch_size
    
    rank = comm.get().get_rank()
    for i in range(num_epochs): 
        crypten.print(f"Epoch {i} in progress:")       
        
        for batch in range(num_batches):
            # define the start and end of the training mini-batch
            start, end = batch * batch_size, (batch + 1) * batch_size
                                    
            # construct CrypTensors out of training examples / labels
            x_train = x_combined_enc[start:end]
            y_batch = labels_one_hot[start:end]
            y_train = crypten.cryptensor(y_batch, requires_grad=True)
            
            # perform forward pass:
            output = model(x_train)
            loss_value = loss(output, y_train)
            
            # set gradients to "zero" 
            model.zero_grad()

            # perform backward pass: 
            loss_value.backward()

            # update parameters
            model.update_parameters(learning_rate)
            
            # Print progress every batch:
            batch_loss = loss_value.get_plain_text()
            crypten.print(f"\tBatch {(batch + 1)} of {num_batches} Loss {batch_loss.item():.4f}")
    

# run_encrypted_training()
