#model:
#one fully connected hidden layer of size 100
#SGD with batch size 128
#learning rate .001
#Peak validation accuracy ~97%

#things to do:
#regularisation
#early stopping
#transformation

import torch.nn as nn
import torch
import numpy as np
from pandas import read_csv
from torch.utils.data import Dataset, DataLoader

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
                    nn.Linear(784, hidden_size),
                    nn.ReLU(),  #if sigmoid results not so good
                    nn.Linear(hidden_size, 10))

    def forward(self, x):
        return self.network(x)

class MNIST(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        else:
            return self.x[index], self.y[index]

def load_data(fname, cut=.8333333):
    df = read_csv(fname)
    x = torch.tensor(df.drop("label", axis=1).values / 255., dtype=torch.float)
    y = torch.tensor(df.label.values, dtype=torch.long)
    divider = int(len(x) * cut)
    train_dataset = MNIST(x=x[:divider], y=y[:divider])
    valid_dataset = MNIST(x=x[divider:], y=y[divider:])
    return train_dataset, valid_dataset

if __name__ == '__main__':
    #parameters
    hidden_size = 100
    learning_rate = .001
    batch_size = 128
    #combination of learning_rate, batch_size = .01 10 does not improve the result - 0.1x accuracy for 10 epochs
    epochs = 80
    #fname = '../data/kaggle-mnist/train.csv'
    fname = '../data/mnist/mnist_train.csv'

    #prepare model
    model = MLP(hidden_size)
    print("model: {}".format(model))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=.0001)  #none of the .1, .01, .001, .0001 l2 regularisation helps with the accuracy
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #prepare data
    print("Loading data...")
    train_dataset, valid_dataset = load_data(fname)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset))

    #train
    #for xbatch, ybatch in get_batch(xtrain, ytrain, batch_size):
    print("Training...")
    max_valid_acc = 0.
    arg_max_valid = 0
    for epoch in range(epochs):
        print("Epoch {}:".format(epoch))
        correct = 0
        optimizer.zero_grad() #Important - the learning would not converge without this. But why? Because otherwise the gradient is accumulated (good for RNN), see https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903
        for x, y in train_loader:
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(output, 1)
            correct += (pred == y).sum().item()
        print('Train accuracy:      {}'.format(correct / len(train_dataset)))
        correct = 0
        for x, y in valid_loader:
            output = model(x)
            _, pred = torch.max(output, 1)
            correct += (pred == y).sum().item()
        valid_acc = correct / len(valid_dataset)
        if max_valid_acc < valid_acc:
            arg_max_valid = epoch
            max_valid_acc = valid_acc
        print('Validation accuracy: {}'.format(valid_acc))

    print('Best validation accuracy {} achieved at epoch {}'.format(max_valid_acc, arg_max_valid))

############################### Old code, not in use ###############################

class DataDealer:
    def __init__(self):
        pass
    
    def load(self, fname, cut=.8):
        df = read_csv(fname)
        x = torch.tensor(df.drop("label", axis=1).values / 255., dtype=torch.float)
        l = torch.tensor(df.label.values, dtype=torch.long)
        size = len(l)
        y = torch.tensor(np.zeros((size, 10)), dtype=torch.long) #torch.zeros?
        y[np.arange(size), l] = 1
        divider = int(size * cut)
        return (x[:divider], y[:divider]), (x[divider:], l[divider:])

def get_batch(xs, ys, batch_size):
    N = len(xs)
    for i in range(0, N, batch_size):
        xbatch = xs[i * batch_size : (i + 1) * batch_size]
        ybatch = ys[i * batch_size : (i + 1) * batch_size]
        yield (xbatch, ybatch)

