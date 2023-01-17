from imports import *

class RandomDataset(Dataset):
    def __init__(self, data_size):
        random.seed(8560)
        x, y = make_blobs(n_samples=data_size, centers=3, n_features=4)

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


def split_data_loaders(n_train, n_test=10000, n_batch=10):
    n_samples = n_train + n_test
    data = RandomDataset(n_samples)
    train_dataset, test_dataset = random_split(data, [n_train, n_test])
    train_loader = DataLoader(train_dataset, batch_size=n_batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=n_batch, shuffle=True)
    return (train_loader, test_loader)







    