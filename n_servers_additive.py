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
from statistics import mean
import crypten.mpc as mpc
import crypten.communicator as comm
import torch.nn as nn
import torch.nn.functional as F




ALICE = 0
BOB = 1



#Define an example network
class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1) 
    
crypten.common.serial.register_safe_class(ExampleNet)

crypten.init()

labels = torch.load('tmp/train_labels.pth')
labels = labels.long()
label_eye = torch.eye(10)
labels_one_hot = label_eye[labels]

@mpc.run_multiprocess(world_size=2)
def run_encrypted_training():
    # Load data:
    x_alice_enc = crypten.load_from_party('tmp/alice_train.pth', src=ALICE)
    x_bob_enc = crypten.load_from_party('tmp/bob_train.pth', src=BOB)
    

    # print("here")
    crypten.print(x_alice_enc.size())
    crypten.print(x_bob_enc.size())
    
    # Combine the feature sets: identical to Tutorial 3
    x_combined_enc = crypten.cat([x_alice_enc, x_bob_enc], dim=2)
    
    # Reshape to match the network architecture
    # x_combined_enc = x_combined_enc.unsqueeze(1)
    # crypten.print(x_combined_enc.size())

    # Initialize a plaintext model and convert to CrypTen model
    pytorch_model = ExampleNet()
    dummy_input = torch.empty((1, 784))
    model = crypten.nn.from_pytorch(pytorch_model, dummy_input)
    model.encrypt()
    # Set train mode
    model.train()
  
    # Define a loss function
    loss_criterion = crypten.nn.CrossEntropyLoss()
    optimizer = crypten.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)
    # Define training parameters
    num_epochs = 10
    batch_size = 1000
    num_batches = x_combined_enc.size(0) // batch_size
    
    rank = comm.get().get_rank()
    for i in range(num_epochs): 
        correct = 0
        crypten.print(f"Epoch {i+1} in progress:")       
        
        for batch in range(num_batches):
            # define the start and end of the training mini-batch
            start, end = batch * batch_size, (batch + 1) * batch_size
                                    
            # construct CrypTensors out of training examples / labels
            x_train = x_combined_enc[start:end].view(-1, 784)
            y_batch = labels_one_hot[start:end]
            y_train = crypten.cryptensor(y_batch, requires_grad=True)
            
            # perform forward pass:
            output = model(x_train)
            loss = loss_criterion(output, y_train)
            
            # set gradients to "zero" 
            model.zero_grad()

            # perform backward pass: 
            loss.backward()

            # update parameters
            optimizer.step()
            
            # Print progress every batch:
            batch_loss = loss.get_plain_text()
            pred = output.get_plain_text().argmax(1)
            labs = labels[start:end]
            crypten.print(f"\tBatch {(batch + 1)} of {num_batches} Loss {batch_loss.item():.4f}")
            # crypten.print(f"Output is {output.get_plain_text().argmax(1)}")
            # crypten.print(f"Label is {labs}")
            # print(f"types = {type(output.get_plain_text())} , {type(y_train.get_plain_text())}")
            # label_eye = torch.eye(10)
            # crypten.print(f'Correct: {(pred == labs).sum()}')
            correct += (pred == labs).sum()
            # crypten.print(f'Correct {correct}')


        accuracy = 100 * correct / 60000
        crypten.print(f"Accuracy = {accuracy:.4f}")
        

torch.set_num_threads(1)
run_encrypted_training()

