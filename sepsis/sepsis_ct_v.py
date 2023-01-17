import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import crypten
import torch
from statistics import mean
from sepsis_dataset import SepsisDataset, split_data_loaders

class Net(crypten.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = crypten.nn.Sequential(
            crypten.nn.Linear(3, 50),
            crypten.nn.ReLU()
        )
        self.layer2 = crypten.nn.Sequential(
            crypten.nn.Linear(50, 50),
            crypten.nn.ReLU()
        )
        self.layer3 = crypten.nn.Sequential(
            crypten.nn.Linear(50, 50),
            crypten.nn.ReLU()
        )
        self.layer4 = crypten.nn.Sequential(
            crypten.nn.Linear(50, 2),
            crypten.nn.LogSoftmax(dim=1)
        )
        self.serv = False

    def forward(self, x):
        if x.shape == torch.Size([10, 3]):
            x = self.layer1(x)
            x = self.layer2(x)
            self.serv = True
        else:
            
            x = self.layer3(x)
            x = self.layer4(x)


        self.serv = not self.serv
        return x


def make_clients(num_clients):
    client_net = []
    for _ in range(num_clients):
        model = Net()
        model.encrypt()
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
        for data in trainset:
            # encrypt the data
            X1, y1 = data
            x1_enc = crypten.cryptensor(X1)
            y1_one_hot = torch.nn.functional.one_hot(y1)
            y1_enc = crypten.cryptensor(y1_one_hot)

            # train first 2 layers using client
            client_output = []
            for h in range(num_clients):
                if h == 0:
                    client_start = time.time()
                    c = client_net[h](x1_enc)
                    client_end = time.time() - client_start
                    client_runtimes.append(client_end)
                else:
                    c = client_net[h](x1_enc)

                client_output.append(c)


            server_start = time.time()
            # transfer network to server
            net = client_net[0]
            # finish last 2 layers in server side
            output = net(client_output[0])
            # send network back to client side to update parameters and repeat
            client_net[0] = net

            print(output.size())
            print(y1.size())



            if(output.size() != y1_enc._tensor.size()):
                continue
            loss = loss_criterion(output, y1_enc)
            net.zero_grad()
            loss.backward()
            optimizer.step()
            server_end = time.time() - server_start
            server_runtimes.append(server_end)

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
        client.decrypt()
        for data in testset:
            X, y = data
            client_output = client(X)
            output = client(client_output)
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 3))


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
    PATH = "models/sepsis_ct_v.pth"
    CLIENTS = 1000
    SIZE = 100000//CLIENTS
    EPOCHS = 5
    print(f'CLIENTS {CLIENTS}, SIZE {SIZE}, EPOCHS {EPOCHS}')

    crypten.init()
    # set up dataset
    dataset = SepsisDataset(SIZE)
    trainset, testset = split_data_loaders(dataset)

    # make clients and loss function / optimizer
    client_net = make_clients(CLIENTS)
    loss_criterion = crypten.nn.CrossEntropyLoss()
    optimizer = crypten.optim.SGD(
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
    # total_runtime_all_epochs : 1064.77686
    # avg_epoch_runtime : 212.95537
    # avg_server_runtime : 0.16444
    # avg_client_runtime : 0.00721
    # avg_runtime : 0.17165
    # Accuracy:  0.926

# CLIENTS 100, SIZE 1000, EPOCHS 5
    # total_runtime_all_epochs : 421.08159
    # avg_epoch_runtime : 84.21632
    # avg_server_runtime : 0.17535
    # avg_client_runtime : 0.00769
    # avg_runtime : 0.18304
    # Accuracy:  0.960

# CLIENTS 500, SIZE 200, EPOCHS 5
    # total_runtime_all_epochs : 348.821
    # avg_epoch_runtime : 69.7642
    # avg_server_runtime : 0.1692
    # avg_client_runtime : 0.00794
    # avg_runtime : 0.17715
    # Accuracy:  0.950

# CLIENTS 1000, SIZE 100, EPOCHS 5
    # total_runtime_all_epochs : 341.39484
    # avg_epoch_runtime : 68.27897
    # avg_server_runtime : 0.16912
    # avg_client_runtime : 0.00855
    # avg_runtime : 0.17776
    # Accuracy:  0.990

