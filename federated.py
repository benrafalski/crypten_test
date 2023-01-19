import torch
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
# import crypten

# %matplotlib inline
plt.rcParams['figure.figsize'] = [5, 5]
# crypten.init()


# get datasets
train_dataset = MNIST('./kaggle/working', train=True, download=True, transform=transforms.ToTensor())
test_dataset = MNIST('./kaggle/working', train=False, download=True, transform=transforms.ToTensor())

train_dataset, dev_dataset = random_split(train_dataset, [int(len(train_dataset) * 0.83), int(len(train_dataset) * 0.17)])

total_train_size = len(train_dataset)
total_test_size = len(test_dataset)
total_dev_size = len(dev_dataset)

# parameters
classes = 10
input_dim = 784

num_clients = 8
rounds = 30
batch_size = 128
epochs_per_client = 3
learning_rate = 2e-2

print(total_train_size, total_dev_size, total_test_size)

# device support for GPU
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader(DataLoader):
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            for batch in self.dl:
                yield to_device(batch, self.device)

        def __len__(self):
            return len(self.dl)

device = get_device()

class FederatedNet(torch.nn.Module):    
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)
        self.track_layers = {'fc1': self.fc1, 'fc2': self.fc2, 'fc3': self.fc3, 'fc4': self.fc4}

    def forward(self, x_batch):
        x = F.relu(self.fc1(x_batch))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

    def forward_client(self, x_batch):
        x = F.relu(self.fc1(x_batch))
        x = F.relu(self.fc2(x))
        # print(f'client out type {type(x)}')
        return x

    def forward_server(self, x):
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
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
    
    # calculates the accuracy for each batch
    def batch_accuracy(self, outputs, labels):
        with torch.no_grad():
            _, predictions = torch.max(outputs, dim=1)
            return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))
    
    def _process_batch(self, batch):
        features, labels = batch
        outputs = self(features)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        accuracy = self.batch_accuracy(outputs, labels)
        return (loss, accuracy)

    # returns the loss and accuracy for each batch
    def _process_client_batch(self, batch):
        images, labels = batch
        images = images.view(-1,784)
        outputs = self.forward_client(images)
        # print(f'client batch type {type(outputs)}')
        return (outputs, labels)

    def _process_server_batch(self, batch):
        output, labels = batch
        # print(f'output type: {type(output)}')
        outputs = self.forward_server(output)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        accuracy = self.batch_accuracy(outputs, labels)
        return (loss, accuracy)
    
    # trains the federated model
    def fit_client(self, dataset, epochs, lr, batch_size=128, opt=torch.optim.SGD):
        dataloader = DeviceDataLoader(DataLoader(dataset, batch_size, shuffle=True), device)
        outputs = []
        labels = []
        for batch in dataloader:
            output, label = self._process_client_batch(batch)
            # print(f'fit client output type {type(output)}')
            outputs.append(output)
            labels.append(label)

        merged_list = [(outputs[i], labels[i]) for i in range(0, len(outputs))]

        return merged_list
    
    def fit_server(self, dataloader, lr, opt=torch.optim.SGD):

        optimizer = opt(self.parameters(), lr)
        history = []
        losses = []
        accs = []
        for batch in dataloader:
            # batch needs to be a tuple
            # print(f'type batch {type(dataloader)}')
            loss, acc = self._process_server_batch(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss.detach()
            losses.append(loss)
            accs.append(acc)
        avg_loss = torch.stack(losses).mean().item()
        avg_acc = torch.stack(accs).mean().item()
        history.append((avg_loss, avg_acc))
        print('Loss = {}, Accuracy = {}'.format(round(history[-1][0], 4), round(history[-1][1], 4)))
        return history

    # evalutates the model
    def evaluate(self, dataset, batch_size=128):
        dataloader = DeviceDataLoader(DataLoader(dataset, batch_size), device)
        losses = []
        accs = []
        with torch.no_grad():
            for batch in dataloader:
                loss, acc = self._process_batch(batch)
                losses.append(loss)
                accs.append(acc)
        avg_loss = torch.stack(losses).mean().item()
        avg_acc = torch.stack(accs).mean().item()
        return (avg_loss, avg_acc)


class Client:
    def __init__(self, client_id, dataset):
        self.client_id = client_id
        self.dataset = dataset
    
    # returns the data size
    def get_dataset_size(self):
        return len(self.dataset)
    
    # returns the client id
    def get_client_id(self):
        return self.client_id
    
    # trains the client model
    def train(self, parameters_dict):
        # define a network
        net = to_device(FederatedNet(), device)
        # sets each layer's weight and bias to the weight and bias from parameter_dict
        net.apply_parameters(parameters_dict)
        # train the client model
        train_history = net.fit_client(self.dataset, epochs_per_client, learning_rate, batch_size)
        # train_history = output, labels
            # output = list of outputs from each batch
            # labels = list of labels from each batch
        # returns this clients parameters
        return (net.get_parameters(), train_history)


# number of images per client
examples_per_client = total_train_size // num_clients
# define a list of client datasets
client_datasets = random_split(train_dataset, [min(i + examples_per_client, total_train_size) - i for i in range(0, total_train_size, examples_per_client)])
# define the clients
clients = [Client('client_' + str(i), client_datasets[i]) for i in range(num_clients)]

# define a global neural network
global_net = to_device(FederatedNet(), device)
history = []
# training
for i in range(rounds):
    print('Start Round {} ...'.format(i + 1))
    # get the weights and biases from the global model
    curr_parameters = global_net.get_parameters()
    print(f'type for params is {curr_parameters}')
    # initialize new weight and biases to 0
    new_parameters = dict([(layer_name, {'weight': 0, 'bias': 0}) for layer_name in curr_parameters])
    # train each client
    client_outs = []
    for client in clients:
        # train the client using the current weights and biases from the global model
        client_parameters, output_list = client.train(curr_parameters)
        client_outs.append(output_list)
        fraction = client.get_dataset_size() / total_train_size
        # save the new weights and biases for each client
        for layer_name in client_parameters:
            new_parameters[layer_name]['weight'] += fraction * client_parameters[layer_name]['weight']
            new_parameters[layer_name]['bias'] += fraction * client_parameters[layer_name]['bias']

    
    for out_list in client_outs:
        global_net.fit_server(out_list, learning_rate)

    # print('{}: Loss = {}, Accuracy = {}'.format(self.client_id, round(train_history[-1][0], 4), round(train_history[-1][1], 4)))

    # update the weights and biases for the global model
    global_net.apply_parameters(new_parameters)
    
    # evaluate the global model
    train_loss, train_acc = global_net.evaluate(train_dataset)
    dev_loss, dev_acc = global_net.evaluate(dev_dataset)
    print('After round {}, train_loss = {}, dev_loss = {}, dev_acc = {}\n'.format(i + 1, round(train_loss, 4), 
            round(dev_loss, 4), round(dev_acc, 4)))
    history.append((train_loss, dev_loss))












