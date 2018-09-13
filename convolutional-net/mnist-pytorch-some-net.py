#model:
#exp10
#~98.85% valid accuracy

#things to do:
#regularisation
#early stopping
#transformation

import torch.nn as nn
import torch
import numpy as np
from pandas import read_csv
from torch.utils.data import Dataset, DataLoader

class SomeNet(nn.Module):
    def __init__(self, hidden_size):
        super(SomeNet, self).__init__()
        self.conv_pool = nn.Sequential(
                    nn.Conv2d(1, 20, 5),
                    nn.MaxPool2d(2),
                    nn.ReLU())
        self.linear = nn.Sequential(
                    nn.Linear(20 * 12 * 12, hidden_size),
                    nn.ReLU(),  #if sigmoid results not so good
                    nn.Linear(hidden_size, 10))

    def forward(self, x):
        x1 = self.conv_pool(x)
        x1 = x1.view(x1.size(0), -1)
        return self.linear(x1)

class MNIST(Dataset):
    def __init__(self, x, y=None):
        self.x = x.unsqueeze(1)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        else:
            return self.x[index], self.y[index]

def load_data(fname, cut=.8333334, gpu=False):
    df = read_csv(fname)
    xarr = df.drop("label", axis=1).values / 255.
    xarr.shape = (len(xarr), 28, 28)
    x = torch.tensor(xarr, dtype=torch.float)
    y = torch.tensor(df.label.values, dtype=torch.long)
    if gpu:
        x = x.cuda()
        y = y.cuda()
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
    gpu = False

    #prepare model
    model = SomeNet(hidden_size)
    if gpu: model.cuda()

    print("model: {}".format(model))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=.0001)  #none of the .1, .01, .001, .0001 l2 regularisation helps with the accuracy
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #prepare data
    print("Loading data...")
    train_dataset, valid_dataset = load_data(fname, gpu=gpu)
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
