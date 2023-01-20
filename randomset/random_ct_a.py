import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import crypten
import time
import torch
from statistics import mean
from random_dataset import split_data_loaders

class Client(crypten.nn.Module):
    def __init__(self, hidden):
        super(Client, self).__init__()

        self.layer1 = crypten.nn.Sequential(
            crypten.nn.Linear(4, hidden),
            crypten.nn.ReLU()
        )
        self.layer2 = crypten.nn.Sequential(
            crypten.nn.Linear(hidden, hidden),
            crypten.nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Global(crypten.nn.Module):
    def __init__(self, models):
        super(Global, self).__init__()
        self.models = models
        self.layer3 = crypten.nn.Sequential(
            crypten.nn.Linear(1000, 50),
            crypten.nn.ReLU()
        )
        self.layer4 = crypten.nn.Sequential(
            crypten.nn.Linear(50, 3),
            crypten.nn.LogSoftmax(dim=1)
        )

    def forward(self, c):   
        start = time.time()   
        client_time = 0  
        client_averages = []
        for i in range(len(c)):
            if i == 0:
                c[i] = self.models[i](c[i])
                client_time = time.time() - start
                client_averages.append(client_time)
            else:
                c[i] = self.models[i](c[i])

        server_time = 0
        start = time.time()
        if(self.encrypted):
            x = crypten.cat(c, dim=1)
        else:
            x = torch.cat(c, dim=1)
        x = self.layer3(x)
        x = self.layer4(x)
        server_time = time.time() - start
        total_time = server_time + client_time
        client_avg = mean(client_averages)
        return x, total_time, server_time, client_avg

def enc_data(data):
    X, y = data
    x_enc = crypten.cryptensor(X)
    y_one_hot = torch.nn.functional.one_hot(y)
    y_enc = crypten.cryptensor(y_one_hot)
    return x_enc, y_enc

def train(num_epochs, train_sets, model, loss_criterion, optimizer):
    epoch_times = []
    server_time = 0
    server_average = []
    client_average = []
    avg_acc = []
    for epoch in range(num_epochs):
        print(f'EPOCH : {epoch+1}') 
        e_start = time.time()
        i=0
        for data in zip(*train_sets): 

            X = []
            y = []
            for d in data:
                a, b = enc_data(d)
                X.append(a)
                y.append(b)

            output, total_time, _, client_avg = model(X)  
            # print(client_avg)
            client_average.append(client_avg)
            
            start = time.time()
            if(output.size() != y[0]._tensor.size()):
                continue
            loss = loss_criterion(output, y[0])  
            model.zero_grad() 
            loss.backward()  
            optimizer.step()
            server_time = time.time() - start
            server_time = server_time + total_time
            server_average.append(server_time)
            
            # with torch.no_grad():
            #     predictions = output.get_plain_text().argmax(1)
            #     acc = (predictions == y).sum() / len(predictions)
            #     avg_acc.append(acc)

        # print(f'Epoch accuracy is {round(mean(avg_acc), 5)}')
        e_time = time.time() - e_start
        epoch_times.append(e_time)
    print(f'Avg Client Time : {mean(client_average)}')
    print(f'Server Time : {mean(server_average)}')
    print(f'Runtime : {sum(epoch_times)}')
    print(f'Average Epoch time : {mean(epoch_times)}')



def evaluate(num_clients, model, clients, test):
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():

        for _i in range(num_clients):
            clients[_i].decrypt()
        model.decrypt()
        for data in zip(*test):
            X = []
            y = []
            for d in data:
                a, b = d
                X.append(a)
                y.append(b) 
            output, _, _, _ = model(X)  
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[0][idx]:
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
    PATH = "models/aggregate_ct.pth"
    CLIENTS = 10
    HIDDENLAYER = 1000//CLIENTS
    EPOCHS = 80
    SIZE = 100000//CLIENTS
    print(f'CLIENTS {CLIENTS}, HIDDEN {HIDDENLAYER}, EPOCHS {EPOCHS}, SIZE {SIZE}')

    crypten.init()
    # data setup
    # sepsis = SepsisDataset(SIZE)
    trainset = []
    test = []
    for _ in range(CLIENTS):
        a, b = split_data_loaders(SIZE)
        trainset.append(a)
        test.append(b)

    clients = []
    for _ in range(CLIENTS):
        model = Client(HIDDENLAYER)
        model.encrypt()
        clients.append(model)

    model = Global(clients)
    model.encrypt()

    loss_criterion = crypten.nn.CrossEntropyLoss()
    optimizer = crypten.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)


    save("models/random_a_ctos.pth", EPOCHS, clients[0], optimizer)
    save("models/random_a_stoc.pth", EPOCHS, model, optimizer)
    train(EPOCHS, trainset, model, loss_criterion, optimizer)
    evaluate(CLIENTS, model, clients, test)
    save(PATH, EPOCHS, model, optimizer)

if __name__ == "__main__":
    main()



# CLIENTS 10, HIDDEN 100, EPOCHS 5, SIZE 10000
    # Avg Client Time : 0.008130154418945313
    # Server Time : 0.32344238902891453
    # Runtime : 1975.9207582473755
    # Average Epoch time : 395.1841516494751
    # Accuracy:  1.0

# CLIENTS 100, HIDDEN 10, EPOCHS 20, SIZE 1000
    # Avg Client Time : 0.006637524962425232
    # Server Time : 1.3116387471860769
    # Runtime : 3923.469391345978
    # Average Epoch time : 196.1734695672989
    # Accuracy:  0.995

# CLIENTS 500, HIDDEN 2, EPOCHS 50, SIZE 200
    # Avg Client Time : 0.007385303258895874
    # Server Time : 5.887607179528954
    # Runtime : 9237.756758451462
    # Average Epoch time : 184.75513516902924
    # Accuracy:  0.401

# CLIENTS 1000, HIDDEN 1, EPOCHS 80, SIZE 100
    # Avg Client Time : 0.00693495899438858
    # Server Time : 11.973492205897465
    # Runtime : 15122.581511735916
    # Average Epoch time : 189.03226889669895
    # Accuracy:  0.637

