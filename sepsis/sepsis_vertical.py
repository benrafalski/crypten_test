from imports import *
from sepsis_dataset import *
from networks import VerticalNet

def train(epochs, model, optimizer, train_loader):
    start = time.time()
    for epoch in range(epochs): 
        print(f'EPOCH {epoch+1}')
        accs = []
        losses = []
        for data in train_loader:  
            loss, acc = model._process_batch(data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss.detach()
            losses.append(loss)
            accs.append(acc)
        avg_acc = mean(accs)
        avg_loss = torch.stack(losses).mean().item()
        print(f'Average Accuracy: {round(avg_acc, 5)}, Average Loss: {round(avg_loss, 5)}')
    print(f'Runtime : {time.time()-start}')


def evaluate(test_loader, model):
    losses = []
    accs = []
    with torch.no_grad():
        for batch in test_loader:
            loss, acc = model._process_batch(batch)
            losses.append(loss)
            accs.append(acc)
    avg_loss = torch.stack(losses).mean().item()
    avg_acc = mean(accs)
    return (avg_loss, avg_acc)


def main():
    CLIENTS = 2
    DATA_SIZE = 100000//CLIENTS
    HIDDEN = 50
    EPOCHS = 5


    sepsis_data = SepsisDataset(DATA_SIZE)
    train_loader, test_loader = split_data_loaders(sepsis_data)

    model = VerticalNet(HIDDEN)
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-6)

    train(EPOCHS, model, optimizer, train_loader)
    print("Evaluating...")
    loss, acc = evaluate(test_loader, model)
    print(f'Accuracy {round(acc, 5)}, Loss {loss}') 

if __name__ == "__main__":
    main()





