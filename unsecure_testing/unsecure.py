from torch.utils.data import random_split
import matplotlib.pyplot as plt
from data import get_device, to_device, SepsisDataset, RandomDataset
from net import FederatedNet
from client import Client
from args import parse_args
import plotext as plt

def main():
    args = parse_args()
    num_clients = args.clients

    if args.sepsis:
        classes = 2
        input_dim = 3

        train_dataset = SepsisDataset(train=True, num_clients=num_clients)
        test_dataset = SepsisDataset(train=False, num_clients=num_clients)

        train_dataset, dev_dataset = random_split(train_dataset, [int(len(train_dataset) * 0.83), int(len(train_dataset) * 0.17)])

        print(f'{len(train_dataset)} {len(dev_dataset)} {len(test_dataset)}')

    elif args.random:
        classes = 3
        input_dim = 4

        train_dataset = RandomDataset(train=True)
        test_dataset = RandomDataset(train=False)

        train_dataset, dev_dataset = random_split(train_dataset, [int(len(train_dataset) * 0.83), int(len(train_dataset) * 0.17)])

        print(f'{len(train_dataset)} {len(dev_dataset)} {len(test_dataset)}')
    
    else:
        print("Error: please specify data set argument value")

    total_train_size = len(train_dataset)
    total_test_size = len(test_dataset)
    total_dev_size = len(dev_dataset)

    # num_clients = args.clients
    rounds = args.epochs
    batch_size = args.batch
    epochs_per_client = 3
    learning_rate = 2e-2

    device = get_device()

    examples_per_client = total_train_size // num_clients
    client_datasets = random_split(train_dataset, [min(i + examples_per_client, 
            total_train_size) - i for i in range(0, total_train_size, examples_per_client)])
    clients = [Client('client_' + str(i), client_datasets[i]) for i in range(num_clients)]


    global_net = to_device(FederatedNet(input_dim, classes), device)
    history = []
    round_acc = []
    for i in range(rounds):
        print('Start Round {} ...'.format(i + 1))
        curr_parameters = global_net.get_parameters()
        new_parameters = dict([(layer_name, {'weight': 0, 'bias': 0}) for layer_name in curr_parameters])
        for client in clients:
            client_parameters = client.train(curr_parameters, device, input_dim, classes, epochs_per_client, learning_rate, batch_size)
            fraction = client.get_dataset_size() / total_train_size
            for layer_name in client_parameters:
                new_parameters[layer_name]['weight'] += fraction * client_parameters[layer_name]['weight']
                new_parameters[layer_name]['bias'] += fraction * client_parameters[layer_name]['bias']
        global_net.apply_parameters(new_parameters)
        
        train_loss, train_acc = global_net.evaluate(train_dataset, device, batch_size)
        dev_loss, dev_acc = global_net.evaluate(dev_dataset, device, batch_size)
        print('After round {}, train_loss = {}, dev_loss = {}, dev_acc = {}\n'.format(i + 1, round(train_loss, 4), 
                round(dev_loss, 4), round(dev_acc, 4)))
        history.append((train_loss, dev_loss))
        round_acc.append(train_acc)

    plt.plot(round_acc)
    plt.theme('matrix')
    plt.plotsize(50, 15)
    plt.show()



if __name__ == '__main__':
    main()

