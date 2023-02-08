import imports


frame = pd.read_csv('sepsis_data/sepsis_survival_primary_cohort.csv')
# print(frame.iloc[1:110204, 0:3])
# print(frame.iloc[1:110204, 3])

class SepsisDataset(Dataset):
    def __init__(self, file_name):
        file_out = pd.read_csv(file_name)
        x = file_out.iloc[1:110001, 0:3].values
        y = file_out.iloc[1:110001, 3].values

        # feature scaling
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y

        # convert to tensors
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


data = SepsisDataset("sepsis_data/sepsis_survival_primary_cohort.csv")
print(len(data))
train_dataset, test_dataset = random_split(data, [int(len(data) * 0.9), int(len(data) * 0.1)])
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

HIDDEN = 50

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, HIDDEN)
        self.fc4 = nn.Linear(HIDDEN, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


net = Net()
print(net)


loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)


def batch_accuracy(output, y):
    with torch.no_grad():
        _, predictions = torch.max(output, dim=1) 
        return (torch.sum(predictions == y).item() / len(predictions))

start = time.time()
for epoch in range(5): 
    print(f'EPOCH {epoch+1}')
    accs = []
    losses = []
    for data in train_loader:  
        X, y = data  
        output = net(X)
        # y = y.reshape(-1, 1)
        # print(output.size()) 
        # print(y.size())
        # print(output) 
        # print(y)
        # y = y.unsqueeze(2)
        loss = loss_criterion(output, y)  
        net.zero_grad() 
        loss.backward()  
        optimizer.step() 
        losses.append(loss)
        acc = batch_accuracy(output, y)
        accs.append(acc)
        

    avg_acc = mean(accs)
    avg_loss = torch.stack(losses).mean().item()
    print(f'avg_acc: {round(avg_acc, 5)}, avg_loss: {avg_loss}')

print(f'Runtime : {time.time()-start}')


correct = 0
total = 0



def _process_batch(model, batch):
    features, labels = batch
    outputs = model(features)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    accuracy = batch_accuracy(outputs, labels)
    return (loss, accuracy)

def evaluate(dataset, model):
    losses = []
    accs = []
    with torch.no_grad():
        for batch in dataset:
            loss, acc = _process_batch(model, batch)
            losses.append(loss)
            accs.append(acc)
    avg_loss = torch.stack(losses).mean().item()
    avg_acc = mean(accs)
    return (avg_loss, avg_acc)

loss, acc = evaluate(test_loader, net)
print(f'Accuracy {round(acc, 5)}, Loss {loss}') 






