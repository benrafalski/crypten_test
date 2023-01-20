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


# crypten.init()
# torch.set_num_threads(1)

# SIZE = 10000
# epochs = 50

# # data stuff
# sepsis = SepsisDataset(SIZE)
# train, test = split_data_loaders(sepsis)

# # print(len(train))

# num_clients = int(sys.argv[1])
# client = []
# for _ in range(num_clients):
#     net = FederatedNet()
#     client.append(net)

    
# dir = "federated_params"
# torch.save(layer1_weights, os.path.join(dir, "layer1_weights.pth"))
# torch.save(layer1_bias, os.path.join(dir, "layer1_bias.pth"))
# torch.save(layer2_weights, os.path.join(dir, "layer2_weights.pth"))
# torch.save(layer2_bias, os.path.join(dir, "layer2_bias.pth"))
@mpc.run_multiprocess(world_size=int(sys.argv[1]))
def secret_share(num_clients, client, train, test, epochs=5, SIZE=1000):

    global_net = FederatedNet()
    
    for epoch in range(epochs):
        avg_acc = []
        losses = []
        batch_num = 0
        crypten.print(f'Starting epoch {epoch+1}')
        for data in train:
            batch_num += 1

            if batch_num % (SIZE/100) == 0:
                crypten.print(f'\tStarting batch {batch_num} of {len(train)}')
            

            global_params = global_net.get_parameters()
            for cl in client:
                cl.apply_parameters(global_params)

            X, y = data

            x = []
            # 1. train 2 layers on client side
            for i in range(num_clients):
                # crypten.print(f'\tTraining client {i}')
                x.append(client[i].forward_client(X))

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

            for cl in client:
                client_params = cl.get_parameters()
                layer1_weights.append(client_params['fc1']['weight'])
                layer1_bias.append(client_params['fc1']['bias'])
                layer2_weights.append(client_params['fc2']['weight'])
                layer2_bias.append(client_params['fc2']['bias'])


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
            
            # 6. backprop
            optimizer = torch.optim.SGD(global_net.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)
            loss_criterion = torch.nn.BCELoss()
            y = torch.unsqueeze(y, 0)
            loss = loss_criterion(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss.detach()

            metric = BinaryAccuracy()
            with torch.no_grad():
                accuracy = metric(output, y)
                avg_acc.append(accuracy)
            
            losses.append(loss)

            # 7. update the parameters for the client
            


        avg_loss = torch.stack(losses).mean().item()
        avg_accuracy = torch.stack(avg_acc).mean().item()
        
        crypten.print(f'Epoch {epoch+1} accuracy: {round(avg_accuracy, 5)}, Loss: {round(avg_loss, 5)}')

    crypten.print('Evaluating global model...')
    avg_loss, avg_acc = global_net.evaluate(test)
    crypten.print(f'avg_loss {avg_loss}, avg_acc {avg_acc}')


# shares = secret_share(num_clients, client, train, test, epochs)
# print(shares)


def main():

    if len(sys.argv) < 4:
        print("Missing Arguments")

    crypten.init()
    torch.set_num_threads(1)


    SIZE = int(sys.argv[2])
    epochs = int(sys.argv[3])

    # data stuff
    sepsis = SepsisDataset(SIZE)
    train, test = split_data_loaders(sepsis)

    num_clients = int(sys.argv[1])
    client = []
    for _ in range(num_clients):
        net = FederatedNet()
        client.append(net)

    shares = secret_share(num_clients, client, train, test, epochs, SIZE)
    print(shares)

if __name__ == "__main__":
    main()



# 100 = 0.79889
