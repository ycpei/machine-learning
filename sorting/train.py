from data import *
from model import *
from torch.utils.data import DataLoader

if __name__ == '__main__':
    hidden_size = 1000
    learning_rate = .001
    batch_size = 64
    epochs = 80
    train_dataset, valid_dataset, labels, label_dict = read_data()
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset))
    #print(train_dataset.shape)
    model = MLP(input_size=3, hidden_size=hidden_size)
    #print("Training...")
    #for epoch in range(epochs):
