import torch.nn as nn
import torch
import numpy as np
from pandas import read_csv

class FC(nn.Module):
    def __init__(self, hidden_size):
        super(FC, self).__init__()
        self.network = nn.Sequential(
                    nn.Linear(784, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 10),
                    nn.Softmax())

    def forward(self, x):
        return self.network(x)

class DataDealer:
    def __init__(self):
        pass
    
    def load(self, fname, cut=.8):
        df = read_csv(fname)
        x = torch.tensor(df.drop("label", axis=1).values / 255.)
        l = torch.tensor(df.label.values)
        size = len(l)
        y = torch.tensor(np.zeros((size, 10))) #torch.zeros?
        y[np.arange(size), l] = 1
        divider = int(size * cut)
        return (x[:divider], y[:divider]), (x[divider:], l[divider:])

def get_batch(xs, ys, batch_size):
    N = len(xs)
    for i in range(0, N, batch_size):
        xbatch = xs[i * batch_size : (i + 1) * batch_size]
        ybatch = ys[i * batch_size : (i + 1) * batch_size]
        yield (xbatch, ybatch)

if __name__ == '__main__':
    #parameters
    hidden_size = 40
    learning_rate = .1
    batch_size = 50

    #prepare model
    model = FC(hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    #prepare data
    dd = DataDealer()
    (xtrain, ytrain), (xval, lval) = dd.load('../data/kaggle-mnist/train.csv')

    #train
    for xbatch, ybatch in get_batch(xtrain, ytrain, batch_size):
        output = model(xbatch)
        loss = criterion(output, ybatch)
        loss.backward()
        optimizer.step()

    #predict
    output = model(xtest)
    lpred = np.max(output, 1)
    print('Accuracy: %f' % np.sum(lpred = lval) / len(lpred))
