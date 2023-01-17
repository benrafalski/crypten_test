import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from statistics import mean
from random_dataset import RandomDataset, split_data_loaders

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 3)


    def forward(self, x):
        if x.shape == torch.Size([10, 4]):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        else:
            x = F.relu(self.fc3(x))
            x = F.log_softmax(self.fc4(x), dim=1) 
        return x 


def make_clients(num_clients):
    client_net = []
    for _ in range(num_clients):
        model = Net()
        model.train()
        client_net.append(model)
    return client_net

def train(num_epochs, trainset, num_clients, client_net, loss_criterion, optimizer):
    client_runtimes = []
    server_runtimes = []
    epoch_runtimes = []
    for epoch in range(num_epochs):
        print(f'Starting epoch #{epoch+1}')
        epoch_start = time.time()
        avg_acc = []
        for data in trainset:
            # encrypt the data
            X1, y1 = data
            # train first 2 layers using client
            client_output = []
            for h in range(num_clients):
                if h == 0:
                    client_start = time.time()
                    c = client_net[h](X1)
                    client_end = time.time() - client_start
                    client_runtimes.append(client_end)
                else:
                    c = client_net[h](X1)

                client_output.append(c)


            server_start = time.time()
            # transfer network to server
            net = client_net[0]
            # finish last 2 layers in server side
            output = net(client_output[0])
            # send network back to client side to update parameters and repeat
            client_net[0] = net

            loss = loss_criterion(output, y1)
            net.zero_grad()
            loss.backward()
            optimizer.step()
            server_end = time.time() - server_start
            server_runtimes.append(server_end)

            with torch.no_grad():
                _, predictions = torch.max(output, dim=1)
                acc = torch.sum(predictions == y1).item() / len(predictions)
                avg_acc.append(acc)

        print(f'Epoch accuracy is {round(mean(avg_acc), 5)}')

        epoch_end = time.time() - epoch_start
        epoch_runtimes.append(epoch_end)

    total_runtime_all_epochs = sum(epoch_runtimes)
    avg_epoch_runtime = mean(epoch_runtimes)
    avg_server_runtime = mean(server_runtimes)
    avg_client_runtime = mean(client_runtimes)
    avg_runtime = avg_server_runtime + avg_client_runtime
    return total_runtime_all_epochs, avg_epoch_runtime, avg_server_runtime, avg_client_runtime, avg_runtime

# testing model accuracy
def evaluate(client, testset):
    correct = 0
    total = 0
    client.eval()

    with torch.no_grad():
        for data in testset:
            X, y = data
            client_output = client(X)
            output = client(client_output)
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 5))


def save(path, epochs, client, optim):
    print(f"Saving model to {path}...")
    state = {
        'epoch': epochs,
        'state_dict': client.state_dict(),
        'optimizer': optim.state_dict(),
    }
    torch.save(state, path)

def main():
    # parameters
    PATH = "models/random_pt_v.pth"
    CLIENTS = 1000
    SIZE = 100000//CLIENTS
    # SIZE = 1000
    EPOCHS = 5
    print(f'CLIENTS {CLIENTS}, SIZE {SIZE}, EPOCHS {EPOCHS}')

    # set up dataset
    trainset, testset = split_data_loaders(SIZE)

    # make clients and loss function / optimizer
    client_net = make_clients(CLIENTS)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        client_net[0].parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)

    # training
    total_runtime_all_epochs, avg_epoch_runtime, avg_server_runtime, avg_client_runtime, avg_runtime = train(
        EPOCHS, trainset, CLIENTS, client_net, loss_criterion, optimizer)

    # runtimes
    print(f"total_runtime_all_epochs : {round(total_runtime_all_epochs, 5)}")
    print(f"avg_epoch_runtime : {round(avg_epoch_runtime, 5)}")
    print(f"avg_server_runtime : {round(avg_server_runtime, 5)}")
    print(f"avg_client_runtime : {round(avg_client_runtime, 5)}")  
    print(f"avg_runtime : {round(avg_runtime, 5)}")

    # evaluate and save the model
    evaluate(client_net[0], testset)
    save(PATH, EPOCHS, client_net[0], optimizer)


if __name__ == "__main__":
    main()


# STATS
# CLIENTS 10, SIZE 10000, EPOCHS 5
    # total_runtime_all_epochs : 1089.17917
    # avg_epoch_runtime : 217.83583
    # avg_server_runtime : 0.17216
    # avg_client_runtime : 0.00721
    # avg_runtime : 0.17938
    # Accuracy:  0.312


