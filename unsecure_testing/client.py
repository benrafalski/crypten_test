from data import to_device
from net import FederatedNet


class Client:
    def __init__(self, client_id, dataset):
        self.client_id = client_id
        self.dataset = dataset
    
    def get_dataset_size(self):
        return len(self.dataset)
    
    def get_client_id(self):
        return self.client_id
    
    def train(self, parameters_dict, device, input_dim, classes, epochs_per_client, learning_rate, batch_size):
        net = to_device(FederatedNet(input_dim, classes), device)
        net.apply_parameters(parameters_dict)
        train_history = net.fit(self.dataset, device, epochs_per_client, learning_rate, batch_size)
        print('{}: Loss = {}, Accuracy = {}'.format(self.client_id, round(train_history[-1][0], 4), round(train_history[-1][1], 4)))
        

        # if(self.client_id == "client_0"):
        #     print(f"{self.client_id} params {net.get_parameters()['fc1']['bias']}")
        
        
        return net.get_parameters()