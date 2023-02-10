import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from statistics import mean
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import BinaryAccuracy
import plotext as plt
import crypten 
import crypten.mpc as mpc
import crypten.communicator as comm

class AccuracyMeter:
    """Measures top-k accuracy of multi-class predictions."""

    def __init__(self, topk=(1,)):
        self.reset()
        self.topk = topk
        self.maxk = max(self.topk)

    def reset(self):
        self.values = []

    def add(self, output, ground_truth):

        # compute predicted classes (ordered):
        _, prediction = output.topk(self.maxk, 1, True, True)
        prediction = prediction.t()

        # store correctness values:
        correct = prediction.eq(ground_truth.view(-1, 10).expand_as(prediction))
        self.values.append(correct[: self.maxk])

    def value(self):
        result = {}
        correct = torch.stack(self.values, 0)
        for k in self.topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            result[k] = correct_k.mul_(100.0 / correct.size(0))
        return result

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = torch.nn.Linear(3, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x
        # if(x.size() == torch.Size([100, 3])):
        #     x = torch.nn.functional.relu(self.fc1(x))
        #     x = torch.nn.functional.relu(self.fc2(x))
        #     return x
        # else:
        #     x = torch.nn.functional.relu(self.fc1(x))
        #     x = torch.nn.functional.relu(self.fc2(x))
        #     x = torch.nn.functional.relu(self.fc3(x))
        #     x = torch.sigmoid(self.fc4(x))
        #     return x
    
crypten.common.serial.register_safe_class(Network)

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


crypten.init()
torch.set_num_threads(1)



# @mpc.run_multiprocess(world_size=3)
def examine_sources():
    # Create a different tensor on each rank
    rank = comm.get().get_rank()
    x = torch.tensor(rank)
    crypten.print(f"Rank {rank}: {x}", in_order=True)
    
    # 
    world_size = comm.get().get_world_size()
    for i in range(world_size):
        x_enc = crypten.cryptensor(x, src=i)
        z = x_enc.get_plain_text()
        
        # Only print from one process to avoid duplicates
        crypten.print(f"Source {i}: {z}")
        
x = examine_sources()

batch_size = 10
train_dataset = SepsisDataset(1000)



loss_criterion = crypten.nn.BCELoss()



# x_alice_enc = crypten.load_from_party('/tmp/alice_train.pth', src=ALICE)


# num_clients = 5
# total_train_size = len(train_dataset)
# examples_per_client = total_train_size // num_clients

# client_datasets = random_split(train_dataset, [min(i + examples_per_client, 
#             total_train_size) - i for i in range(0, total_train_size, examples_per_client)])

# client_datasets = [DataLoader(c, batch_size=batch_size, shuffle=True) for c in client_datasets]


# # features = torch.cat([torch.cat([X for X, y in dataset], dim=0) for dataset in client_datasets], dim=0)
# # labels = torch.cat([torch.cat([y for X, y in dataset], dim=0) for dataset in client_datasets], dim=0)


num_features = 1000//5

num_clients = 5
features = torch.cat([torch.load(f'client_data_sepsis/features/client{i}_features.pth') for i in range(num_clients)], dim=0)
labels = torch.cat([torch.load(f'client_data_sepsis/labels/client{i}_labels.pth') for i in range(num_clients)], dim=0)


# features = [torch.load(f'client_data_sepsis/features/client{i}_features.pth') for i in range(num_clients)]
# labels = [torch.load(f'client_data_sepsis/labels/client{i}_labels.pth') for i in range(num_clients)]
   

def train_multiparty():
    net = Network()
    net = crypten.nn.from_pytorch(net, torch.empty(64, 3))
    net.encrypt()
    net.train()

    epoch_accuracies = []
    for epoch in range(5):
        crypten.print(f'\nEPOCH : {epoch+1}')
        acc = []
        batch_num = 0
        for i in range(0, num_features, batch_size):
            x_batch = features[i:(i+batch_size)]
            y_batch = labels[i:(i+batch_size)]
            batch_num += 1

            if batch_num % ((num_features//10)//10) == 0:
                crypten.print(f'\tStarting batch {batch_num} of {num_features//10}')

            X = x_batch
            X = crypten.cryptensor(X)
            y = y_batch.unsqueeze(dim=0)
            y = crypten.cryptensor(y) 

            output = net(X)
            output = output.view(-1, 10)

            net.decrypt()
            before = net._modules['fc2.bias'].data
            net.encrypt()

            # for param in net.parameters():
            #     crypten.print(f'param {param}')
            
            loss = loss_criterion(output, y)  
            net.zero_grad() 
            loss.backward()  
            net.update_parameters(0.001)
            
            metric = BinaryAccuracy()
            with torch.no_grad():
                avg_acc = metric(output.get_plain_text(), y.get_plain_text())
            
            acc.append(avg_acc)

            net.decrypt()
            after = net._modules['fc2.bias'].data
            net.encrypt()

            crypten.print(torch.equal(after, before))

        epoch_acc = torch.stack(acc).mean().item()
        epoch_accuracies.append(epoch_acc)
        crypten.print(f'Epoch accuracy: {epoch_acc}', in_order=True)

train_multiparty()

def train_multiparty_new():
    epoch_accuracies = []
    for epoch in range(5):
        crypten.print(f'\nEPOCH : {epoch+1}')
        acc = []
        batch_num = 0   
        for i in range(0, num_features, batch_size):
            for idx in range(len(features)):
                x_batch = features[idx][i:(i+batch_size)]
                y_batch = labels[idx][i:(i+batch_size)]
                batch_num += 1

                if batch_num % ((num_features//10)//10) == 0:
                    crypten.print(f'\tStarting batch {batch_num} of {num_features//10}')

                X = x_batch
                X = crypten.cryptensor(X)
                y = y_batch.unsqueeze(dim=0)
                y = crypten.cryptensor(y) 

                output = net(X)
                output = output.view(-1, 10)
                
                loss = loss_criterion(output, y)  
                net.zero_grad() 
                loss.backward()  
                net.update_parameters(0.001)
                
                metric = BinaryAccuracy()
                with torch.no_grad():
                    avg_acc = metric(output.get_plain_text(), y.get_plain_text())
                
                acc.append(avg_acc)

            epoch_acc = torch.stack(acc).mean().item()
            epoch_accuracies.append(epoch_acc)
            crypten.print(f'Epoch accuracy: {epoch_acc}', in_order=True)




def train_multiparty_new():
    epoch_accuracies = []
    for epoch in range(5):
        crypten.print(f'\nEPOCH : {epoch+1}')
        acc = []
        batch_num = 0
        for i in range(0, num_features, batch_size):
            x_batch = features[i:(i+batch_size)]
            y_batch = labels[i:(i+batch_size)]
            batch_num += 1

            if batch_num % ((num_features//10)//10) == 0:
                crypten.print(f'\tStarting batch {batch_num} of {num_features//10}')

            X = x_batch
            X = crypten.cryptensor(X)
            y = y_batch.unsqueeze(dim=0)
            y = crypten.cryptensor(y) 


            print(X.size())
            

            output = net(X)
            output = output.view(-1, 10)

            return
            
            loss = loss_criterion(output, y)  
            net.zero_grad() 
            loss.backward()  
            net.update_parameters(0.001)
            
            metric = BinaryAccuracy()
            with torch.no_grad():
                avg_acc = metric(output.get_plain_text(), y.get_plain_text())
            
            acc.append(avg_acc)

        epoch_acc = torch.stack(acc).mean().item()
        epoch_accuracies.append(epoch_acc)
        crypten.print(f'Epoch accuracy: {epoch_acc}', in_order=True)

# train_multiparty_new()
# train = client_datasets[0]
# epoch_accuracies = []
# for epoch in range(5):
#     print(f'\nEPOCH : {epoch+1}')
#     acc = []
#     batch_num = 0
#     for X, y in client_datasets[0]:

#         batch_num += 1

#         if batch_num % (len(train)//10) == 0:
#             print(f'\tStarting batch {batch_num} of {len(train)}')

#         X = crypten.cryptensor(X)
#         y = torch.unsqueeze(y, 0)
#         y_enc = crypten.cryptensor(y) 


#         output = net(X)
#         output = output.view(-1, 10)
        
#         loss = loss_criterion(output, y_enc)  
#         net.zero_grad() 
#         loss.backward()  
#         net.update_parameters(0.001)


#         metric = BinaryAccuracy()
#         with torch.no_grad():
#             avg_acc = metric(output.get_plain_text(), y)
        
#         acc.append(avg_acc)

#     epoch_acc = torch.stack(acc).mean().item()
#     epoch_accuracies.append(epoch_acc)
#     print(f'Epoch accuracy: {epoch_acc}')


# EPOCH : 1
# acc = 0.30000001192092896
# acc = 0.699999988079071
# acc = 0.30000001192092896
# acc = 0.5
# acc = 0.6000000238418579
# acc = 0.800000011920929


num_clients = 5
num_features = 1000
@mpc.run_multiprocess(world_size=2)
def train_multiparty2():
    rank = comm.get().get_rank()
    crypten.print(f'rank {rank}')
    # batch_size = 10

    # net = Network()
    # net = crypten.nn.from_pytorch(net, torch.empty(64, 3))
    # net.encrypt()
    # net.train()

    # loss_criterion = crypten.nn.BCELoss()
    epoch_accuracies = []
    optimizer = crypten.optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)
    for epoch in range(5):
        crypten.print(f'\nEPOCH : {epoch+1}')
        acc = []
        batch_num = 0
        for i in range(0, num_features, batch_size):
            x_batch = features[i:(i+batch_size)]
            y_batch = labels[i:(i+batch_size)]

            batch_num += 1

            if batch_num % ((num_features//10)//10) == 0:
                crypten.print(f'\tStarting batch {batch_num} of {num_features//10}')

            X = crypten.cryptensor(x_batch)
            y = torch.unsqueeze(y_batch, 0)
            y_enc = crypten.cryptensor(y)
            
            output = net(X)
            output = output.view(-1, 10)

            loss = loss_criterion(output, y_enc)  
            net.zero_grad() 
            loss.backward()  
            # net.update_parameters(0.001)
            optimizer.step()


            # crypten.print(f'features = {x_batch}')
            # crypten.print(f'labels = {y_batch}')

            # net.decrypt()

            # crypten.print(net.fc1)
               

            # meter = AccuracyMeter()
            # meter.add(output.get_plain_text(), y)
            # avg_acc = meter.value()[1]
            
            metric = BinaryAccuracy()
            with torch.no_grad():
                avg_acc = metric(output.get_plain_text(), y)



            crypten.print(f'acc = {avg_acc}')

            # return 

            # avg_acc = compute_accuracy(output, y)

            acc.append(avg_acc)

        epoch_acc = torch.stack(acc).mean().item()
        epoch_accuracies.append(epoch_acc)
        crypten.print(f'Epoch accuracy: {epoch_acc}  {epoch_acc*2}')


# train_multiparty2()


