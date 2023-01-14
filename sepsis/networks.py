from imports import *

class VerticalNet(nn.Module):
    def __init__(self, hidden):
        super(VerticalNet, self).__init__()
        self.fc1 = nn.Linear(3, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

    def batch_accuracy(self, output, y):
        with torch.no_grad():
            _, predictions = torch.max(output, dim=1) 
            return (torch.sum(predictions == y).item() / len(predictions))

    def _process_batch(self, batch):
        features, labels = batch
        outputs = self(features)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        accuracy = self.batch_accuracy(outputs, labels)
        return (loss, accuracy)