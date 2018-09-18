from pandas import read_csv
import numpy as np

def shift_left(row, size=28):
    row = np.append(row[1:], 0)
    for i in range(size - 1, size * (size - 1), size):
        row[i] = 0
    return row

def shift_right(row, size=28):
    row = np.append([0], row[:-1])
    for i in range(size, size * size, size):
        row[i] = 0
    return row

def shift_down(row, size=28):
    return np.append(np.zeros(size, dtype=int), row[:size * (size - 1)])

def shift_up(row, size=28):
    return np.append(row[size:], np.zeros(size, dtype=int))

def idt(row, size=28):
    return row

def transform(ifname, ofname, size=28):
    of = open(ofname, 'w')
    header = ["label"] + ["pixel" + str(i) for i in range(size * size)]
    of.write(",".join(header) + "\n")

    df = read_csv(ifname)
    xs = df.drop("label", axis=1).values
    ys = df.label.values
    for i, (x, y) in enumerate(zip(xs, ys)):
        if (i % 1000 == 0): print(i)
        #for f in [idt, shift_left, shift_right, shift_up, shift_down]:
        for f in [shift_up]:
            newimage = np.append([y], f(x, size=size))
            of.write(",".join([str(i) for i in newimage]) + "\n")

def test():
    x = np.arange(9)
    for f in [idt, shift_left, shift_right, shift_up, shift_down]:
        print(f(x, size=3))

#test()
#transform("mnist_train.csv", "mnist_train_transformed.csv")
#transform("mnist_train.csv", "mnist_train_transformed_left.csv")
transform("mnist_train.csv", "mnist_train_transformed_up.csv")
