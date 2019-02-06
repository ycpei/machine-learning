#generate data:
#input: 3 numbers
#output: 3-permutation

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SortData(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.y is None: return self.x[index]
        else:
            return self.x[index], self.y[index]

def get_perms(a, b):
    """compare a and b and output array of permutations
    """
    m, n = a.shape
    out = ['' for _ in range(m)]
    for i in range(m):
        for j in range(n):
            out[i] += str(np.searchsorted(b[i], a[i][j]))
    return np.array(out)

def gen_data(m=1000, n=3):
    a = np.random.rand(m, n)
    a -= .5
    b = np.sort(a)
    labels = get_perms(a, b)
    return a, labels

def save_data(x, y, outf='data.csv'):
    _, n = x.shape
    header1 = ['num' + str(j) for j in range(n)]
    df1 = pd.DataFrame(x, columns=header1)
    df2 = pd.DataFrame(y, columns=['perm'])
    #print(df2)
    pd.concat([df1, df2], axis=1).to_csv(outf, index=False)
    return

def read_data(inpf='data.csv', cut=.8):
    df = pd.read_csv(inpf, converters={"perm": str})
    x = torch.tensor(df.iloc[:,:-1].values, dtype=torch.float)
    y_str = df.iloc[:, -1].values
    labels = sorted(list(set(y_str)))
    label_dict = {lab: i for i, lab in enumerate(labels)}
    y = torch.tensor([label_dict[lab] for lab in y_str], dtype=torch.int)
    divider = int(len(x) * cut)
    train_dataset = SortData(x=x[:divider], y=y[:divider])
    valid_dataset = SortData(x=x[divider:], y=y[divider:])
    return train_dataset, valid_dataset, labels, label_dict
