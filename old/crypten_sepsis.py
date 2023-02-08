import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch.nn as nn
import torch.nn.functional as F
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm 
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torchmetrics.classification import BinaryAccuracy
import sys
import time
from statistics import mean
import plotext as plt
import logging

# mpc_additive_merging = [0.40, 0.58, 0.65, 0.72, 0.73, 0.9, 0.85, 0.89, 0.96, 0.92, 0.94, 0.95, 0.91, 0.96, 0.96, 0.94, 0.88, 0.9, 0.9, 0.93]
# mpc_additive_merging = [acc*100 for acc in mpc_additive_merging]

# start epoch
# run a batch for a client
# aggregate everything
# finish running batch on server
# 


class TestingDataset(Dataset):
    def __init__(self):
        file_out = pd.read_csv('sepsis/sepsis_data/sepsis_survival_study_cohort.csv')
        x = file_out.iloc[1:19001, 0:3].values
        y = file_out.iloc[1:19001, 3].values

        # feature scaling
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y

        # convert to tensors
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class ClientDataset(Dataset):
    def __init__(self, client_num, file_out, data_per_client):
        self.file_out = file_out
        x = self.file_out.iloc[client_num*data_per_client:(client_num+1)*data_per_client, 0:3].values
        y = self.file_out.iloc[client_num*data_per_client:(client_num+1)*data_per_client, 3].values

        # feature scaling
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y

        # convert to tensors
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]



def split_data_loaders(data):
    train_dataset, test_dataset = random_split(data, [int(len(data) * 0.9), int(len(data) * 0.1)])
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    return (train_loader, test_loader)

class FederatedNet(nn.Module):    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 5)
        self.fc4 = nn.Linear(5, 1)
        self.track_layers = {'fc1': self.fc1, 'fc2': self.fc2, 'fc3': self.fc3, 'fc4': self.fc4}

    def forward(self, x_batch):
        x = F.relu(self.fc1(x_batch))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x).view(-1, 10))
        return x



    def evaluate(self, dataset):
        losses = []
        accs = []
        loss_criterion = torch.nn.BCELoss()
        metric = BinaryAccuracy()
        with torch.no_grad():
            for batch in dataset:
                X, y = batch
                server_out = self(X)
                y = torch.unsqueeze(y, 0)
                loss = loss_criterion(server_out, y)
                with torch.no_grad():
                    acc = metric(server_out, y)
                losses.append(loss)
                accs.append(acc)
        avg_loss = torch.stack(losses).mean().item()
        avg_acc = torch.stack(accs).mean().item()
        return (avg_loss, avg_acc)

    
# dir = "federated_params"
# torch.save(layer1_weights, os.path.join(dir, "layer1_weights.pth"))
# torch.save(layer1_bias, os.path.join(dir, "layer1_bias.pth"))
# torch.save(layer2_weights, os.path.join(dir, "layer2_weights.pth"))
# torch.save(layer2_bias, os.path.join(dir, "layer2_bias.pth"))
@mpc.run_multiprocess(world_size=int(sys.argv[1]))
def secret_share(num_clients, client, client_dataset, test, epochs=1, SIZE=1000):

    global_net = FederatedNet()
    
    global_time_start = time.time()
    epoch_times = []
    epoch_accuracy = []
    for epoch in range(epochs):
        epoch_time_start = time.time()
        avg_acc = []
        losses = []
        batch_num = 0

        client_times_avg = []
        server_times_avg = []

        crypten.print(f'Starting epoch {epoch+1}')
        for dataset in zip(*client_dataset):

            batch_num += 1

            if batch_num % (SIZE//100) == 0:
                crypten.print(f'\tStarting batch {batch_num} of {SIZE//10}')

            
            X = [a for a, _ in dataset]
            y = [a for _, a in dataset]


            client_times = []

            # 1. train 2 layers on client side
            # for i in range(num_clients):
            #     client_i_start = time.time()
            #     X[i] = crypten.cryptensor(X[i], src=i)
            #     client_i_end = time.time()-client_i_start
            #     client_times.append(client_i_end)

            for i in range(num_clients):
                client_i_start = time.time()
                X[i] = crypten.cryptensor(X[i], src=0)
                # torch.save("layer1_weights", os.path.join("./", "layer1_weights.pth")) 
                client_i_end = time.time()-client_i_start
                client_times.append(client_i_end)

            
            # 2. grab the weights and biases from each client and secret share them with the server
            

            avg_client_time = mean(client_times)
            client_times_avg.append(avg_client_time)
            
            # 3. average each weight and bias recieved from the client
            

            
            # sum_clients = F.relu(sum_clients)

            server_time_start = time.time()

            optimizer = torch.optim.SGD(global_net.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)
            loss_criterion = torch.nn.BCELoss()
            y = [torch.unsqueeze(label, 0) for label in y]


            output = []
            for i in range(len(X)):
                output.append(global_net(X[i].get_plain_text()))
                loss = loss_criterion(output[i], y[i])
                losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss.detach()
               


            metric = BinaryAccuracy()
            with torch.no_grad():
                avg_acc = [metric(output[i], y[i]) for i in range(num_clients)]


            

            # for l in loss:
            #     losses.append(l)
            server_time_end = time.time() - server_time_start
            server_times_avg.append(server_time_end)
        
        avg_loss = torch.stack(losses).mean().item()
        avg_accuracy = torch.stack(avg_acc).mean().item()
        epoch_accuracy.append(avg_accuracy)
        epoch_time_current = time.time() - epoch_time_start
        epoch_times.append(epoch_time_current)
        crypten.print(f'Epoch {epoch+1} accuracy: {avg_accuracy}, Loss: {round(avg_loss, 5)}, Time: {round(epoch_time_current, 5)}')
        crypten.print(f'Avg server time: {mean(server_times_avg)}, Avg client time: {mean(client_times_avg)}')

    crypten.print(f'Total Runtime is {time.time() - global_time_start}')
    crypten.print('Evaluating global model...')
    avg_loss, avg_acc = global_net.evaluate(test)
    crypten.print(f'avg_loss {avg_loss}, avg_acc {avg_acc}')

    if comm.get().get_rank() == 0:
        plt.plot(epoch_accuracy)
        plt.theme('matrix')
        plt.plotsize(50, 15)
        plt.show()
    

# args 
# 1 = num clients
# 2 = epochs
# 3 = size of dataset

def main():
    crypten.init()
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Missing Argument")


    num_clients = int(sys.argv[1])
    epochs = int(sys.argv[2])
    data_per_client = int(sys.argv[3])

    # num_clients = 1000


    # make clients and datasets
    file_out = pd.read_csv('sepsis/sepsis_data/sepsis_survival_primary_cohort.csv')
    file_out.sample(frac=1)

    test_dataset = TestingDataset()
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    client_datasets = [DataLoader(ClientDataset(i, file_out, data_per_client), batch_size=10, shuffle=True) for i in range(num_clients)]
    clients = [FederatedNet() for _ in range(num_clients)]

    secret_share(num_clients, clients, client_datasets, test_loader, epochs, data_per_client)


    

if __name__ == "__main__":
    main()

# train first 2 layers on the client side
# secret share the output to the servers
# train the last 2 layers on the server side
# send the output to each client
# each client updates their local models -> for this step, this is the backward pass
# each client secret shares their weights and bias to the servers
# the servers then update the global model
# each client then updates their local model with the global params
# ... problem: servers and clients need to send data back and forth twice each epoch

# train first 2 layers on the client side
# clients secret share the output and the weights and biases to the servers
# update the global model with the aggregated client weight and biases
# train the last 2 layers on the server side
# send the output and the global weight and biases to each client
# each client then updates their local model with the global params
# each client updates their local models -> for this step, this is the backward pass
# ... problem: when the clients share their weight and bias in step 3
# ... the weights and biases will be unchanged since the last round.
# ... I realized this when Manazir told me that the weights and biases only 
# ... get updated when the backward pass is performed.
# ... Thus, if the this method is used then training will be performed on the old
# ... since the parameter updates will be performed out of order