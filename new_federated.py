import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch.nn as nn
import torch.nn.functional as F
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm 


class FederatedNet(torch.nn.Module):    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)
        self.track_layers = {'fc1': self.fc1, 'fc2': self.fc2, 'fc3': self.fc3, 'fc4': self.fc4}

    def forward_client(self, x_batch):
        # x = F.relu(self.fc1(x_batch))
        # x = F.relu(self.fc2(x))
        # return x
        pass

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

crypten.init()
torch.set_num_threads(1)
# 1. train 2 layers on client side
num_clients = 2
client = []
for _ in range(num_clients):
    net = FederatedNet()
    client.append(net)

for i in range(num_clients):
    client[i].forward_client(0)
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

# dir = "federated_params"
# torch.save(layer1_weights, os.path.join(dir, "layer1_weights.pth"))
# torch.save(layer1_bias, os.path.join(dir, "layer1_bias.pth"))
# torch.save(layer2_weights, os.path.join(dir, "layer2_weights.pth"))
# torch.save(layer2_bias, os.path.join(dir, "layer2_bias.pth"))
@mpc.run_multiprocess(world_size=num_clients)
def secret_share(layer1_weights, layer1_bias, layer2_weights, layer2_bias):

    global_net = FederatedNet()

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

    

    for layer_name in client_parameters:
        new_parameters[layer_name]['weight'] += client_parameters[layer_name]['weight']
        new_parameters[layer_name]['bias'] += client_parameters[layer_name]['bias']

    # 5. continue training the global model for the last 2 layers
    # 6. do backprop
    # 7. update the parameters for the client


# print(f'Before : {layer1_weights}')
shares = secret_share(layer1_weights, layer1_bias, layer2_weights, layer2_bias)
print(shares)



# 8. repeat
