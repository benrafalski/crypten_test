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


class SepsisDataset(Dataset):
    def __init__(self, data_size):
        file_out = pd.read_csv('sepsis/sepsis_data/sepsis_survival_primary_cohort.csv')
        x = file_out.iloc[1:(data_size+1), 0:3].values
        y = file_out.iloc[1:(data_size+1), 3].values

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
    def __init__(self, x_train, y_train):
        # convert to tensors
        self.X_train = x_train
        self.y_train = y_train

    def get_batch(self, idx, batch_size=10):
        X = self.X_train[idx*batch_size:(idx+1)*batch_size]
        y = self.y_train[idx*batch_size:(idx+1)*batch_size]
        return (X, y)

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

    def forward_client(self, x_batch):
        x = F.relu(self.fc1(x_batch))
        x = F.relu(self.fc2(x))
        return x

    def forward_server(self, x):
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x).view(-1, 10))
        return x

    # return the dictionary of the layers
    def get_track_layers(self):
        return self.track_layers
    
    # sets each layer's weight and bias to the weight and bias given as argument
    def apply_parameters(self, parameters_dict):
        with torch.no_grad():
            for layer_name in parameters_dict:
                self.track_layers[layer_name].weight.data *= 0
                self.track_layers[layer_name].bias.data *= 0
                self.track_layers[layer_name].weight.data += parameters_dict[layer_name]['weight']
                self.track_layers[layer_name].bias.data += parameters_dict[layer_name]['bias']

    def apply_parameters_server(self, parameters_dict):
        with torch.no_grad():
            layers = ['fc1', 'fc2']
            for layer_name in layers:
                self.track_layers[layer_name].weight.data *= 0
                self.track_layers[layer_name].bias.data *= 0
                self.track_layers[layer_name].weight.data += parameters_dict[layer_name]['weight'].get_plain_text()
                self.track_layers[layer_name].bias.data += parameters_dict[layer_name]['bias'].get_plain_text()
    
    
    # returns a parameter dictionary for each layer
    # dictionary is of the form {"weight": w, "bias": b}
    def get_parameters(self):
        parameters_dict = dict()
        for layer_name in self.track_layers:
            parameters_dict[layer_name] = {
                'weight': self.track_layers[layer_name].weight.data, 
                'bias': self.track_layers[layer_name].bias.data
            }
        return parameters_dict

    def evaluate(self, dataset):
        losses = []
        accs = []
        loss_criterion = torch.nn.BCELoss()
        metric = BinaryAccuracy()
        with torch.no_grad():
            for batch in dataset:
                X, y = batch
                client_out = self.forward_client(X)
                server_out = self.forward_server(client_out)

                y = torch.unsqueeze(y, 0)
                loss = loss_criterion(server_out, y)

                with torch.no_grad():
                    acc = metric(server_out, y)

                losses.append(loss)
                accs.append(acc)
        avg_loss = torch.stack(losses).mean().item()
        avg_acc = torch.stack(accs).mean().item()
        return (avg_loss, avg_acc)

class Client:
    def __init__(self, client_id, dataset, net):
        self.client_id = client_id
        self.dataset = dataset
        self.net = net
    
    # returns the data size
    def get_dataset_size(self):
        return len(self.dataset)
    
    # returns the client id
    def get_client_id(self):
        return self.client_id

    def train(self, global_params, data_idx):
        self.net.apply_parameters(global_params)
        X, y = self.dataset.get_batch(data_idx, batch_size)
        x = self.net.forward_client(X)
        return (x, y, self.net.get_parameters())



crypten.init()
torch.set_num_threads(1)
num_clients = int(sys.argv[1])

SIZE = 1000
client_data_size = SIZE//num_clients
batch_size = 10

# data stuff
sepsis = SepsisDataset(SIZE)

client_datasets = [ClientDataset(sepsis.X_train[i*(client_data_size):(i+1)*client_data_size], sepsis.y_train[i:i+client_data_size]) for i in range(num_clients)]

train, test = split_data_loaders(sepsis)

print(len(client_datasets[0]))

client = [Client('client_' + str(i), client_datasets[i], FederatedNet()) for i in range(num_clients)]


@mpc.run_multiprocess(world_size=num_clients)
def secret_share():

    global_net = FederatedNet()

    epochs = 50
    
    for epoch in range(epochs):
        avg_acc = []
        losses = []
        for idx in range(client_data_size//batch_size):
            


            global_params = global_net.get_parameters()
            # for cl in client:
            #     cl.apply_parameters(global_params)

            x = []
            y = []
            client_params = []
            # 1. train 2 layers on client side
            for i in range(num_clients):
                output, labels, params = client[i].train(global_params, idx)
                x.append(output)
                y.append(labels)
                client_params.append(params)

            # for i in range(num_clients):
            #     x.append(client[i].forward_client(X))

            sum_clients = x[0]
            for i in range(1, num_clients):
                sum_clients += x[i]

            sum_clients /= num_clients
            sum_clients = F.relu(sum_clients)

            # 2. grab the weights and biases from each client and secret share them with the server
            layer1_weights = []
            layer1_bias = []
            layer2_weights = []
            layer2_bias = []

            for cl in client_params:
                layer1_weights.append(cl['fc1']['weight'])
                layer1_bias.append(cl['fc1']['bias'])
                layer2_weights.append(cl['fc2']['weight'])
                layer2_bias.append(cl['fc2']['bias'])

            for i in range(num_clients):
                layer1_weights[i] = crypten.cryptensor(layer1_weights[i], src=i)
                layer1_bias[i] = crypten.cryptensor(layer1_bias[i], src=i)
                layer2_weights[i] = crypten.cryptensor(layer2_weights[i], src=i)
                layer2_bias[i] = crypten.cryptensor(layer2_bias[i], src=i)

            # 3. average each weight and bias recieved from the client
            layer1_weights_total = layer1_weights[0]
            layer1_bias_total = layer1_bias[0]
            layer2_weights_total = layer2_weights[0]
            layer2_bias_total = layer2_bias[0]

            for i in range(1, num_clients):
                layer1_weights_total += layer1_weights[i]
                layer1_bias_total += layer1_bias[i]
                layer2_weights_total += layer2_weights[i]
                layer2_bias_total += layer2_bias[i]

            layer1_weights_total /= comm.get().get_world_size()
            layer1_bias_total /= comm.get().get_world_size()
            layer2_weights_total /= comm.get().get_world_size()
            layer2_bias_total /= comm.get().get_world_size()

            # 4. update the global weight and biases with the averaged parameters
            curr_parameters = global_net.get_parameters()
            new_parameters = dict([(layer_name, {'weight': 0, 'bias': 0}) for layer_name in curr_parameters])

            client_parameters = dict([(layer_name, {'weight': 0, 'bias': 0}) for layer_name in curr_parameters])

            client_parameters['fc1']['weight'] = layer1_weights_total
            client_parameters['fc1']['bias'] = layer1_bias_total
            client_parameters['fc2']['weight'] = layer2_weights_total
            client_parameters['fc2']['bias'] = layer2_bias_total

            for layer_name in client_parameters:
                new_parameters[layer_name]['weight'] = client_parameters[layer_name]['weight']
                new_parameters[layer_name]['bias'] = client_parameters[layer_name]['bias']

            global_net.apply_parameters_server(new_parameters)
            
            # 5. continue training the global model for the last 2 layers
            output = global_net.forward_server(sum_clients)
            
            # crypten.print(global_params)
            global_params = global_net.get_parameters()
            # crypten.print(global_params)

            for i in range(num_clients):
                crypten.print(f'before {client[i].net.get_parameters()}')
                client[i].net.apply_parameters(global_params)
                crypten.print(f'after {client[i].net.get_parameters()}')

            # 6. send back to clients and do backprop
            optimizer = torch.optim.SGD(global_net.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)
            loss_criterion = torch.nn.BCELoss()
            metric = BinaryAccuracy()


            outs = []
            for i in range(num_clients):
                outs.append(output)


            for i in range(num_clients):
                y[i] = torch.unsqueeze(y[i], 0)
                loss = loss_criterion(outs[i], y[i])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss.detach()
                with torch.no_grad():
                    accuracy = metric(outs[i], y[i])
                    avg_acc.append(accuracy)
                
                losses.append(loss)

            # 7. update the parameters for the client
            
            


        avg_loss = torch.stack(losses).mean().item()
        avg_accuracy = torch.stack(avg_acc).mean().item()
        
        crypten.print(f'Epoch {epoch+1} accuracy: {round(avg_accuracy, 5)}, Loss: {round(avg_loss, 5)}')

    crypten.print('Evaluating global model...')
    avg_loss, avg_acc = global_net.evaluate(test)
    crypten.print(f'avg_loss {avg_loss}, avg_acc {avg_acc}')



# print(f'Before : {layer1_weights}')
shares = secret_share()
print(shares)



# 8. repeat

