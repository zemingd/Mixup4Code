from torch_geometric.data import Data,DataLoader
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import random
import copy
import numpy as np
from torch_geometric.nn import global_max_pool
import torch.nn.functional as F

def main():
    data_buggy = []
    data_fixed = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(100):
        edge_index = torch.tensor(Dataset_Buggy_Edge[i], dtype=torch.long)
        x = torch.tensor(Dataset_Buggy_Node[i]), dtype=torch.float)
        y = torch.tensor([0], dtype=torch.long)
        y = torch.nn.functional.one_hot(y, 2)
        data_buggy.append(Data(x=x, edge_index=edge_index.t().contiguous(), y=y))

    for i in range(100):
        edge_index = torch.tensor(Dataset_Fixed_Edge[i], dtype=torch.long)
        x = torch.tensor(Dataset_Fixed_Node[i]), dtype=torch.float)
        y = torch.tensor([1], dtype=torch.long)
        y = torch.nn.functional.one_hot(y, 2)
        data_fixed.append(Data(x=x, edge_index=edge_index.t().contiguous(), y=y))

    data = data_fixed + data_buggy
    random.shuffle(data)

    train_dataset = data[i:]
    test_dataset = data[:i]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=1, shuffle=True)


    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(Vertex_feature_dim, 160)
            self.conv2 = GATConv(160,160)
            self.linear = torch.nn.Linear(160, Num_class)
            pass

        def forward(self, x,batch,edge_index):
            out = self.conv1(x, edge_index)
            max_out = global_max_pool(out,batch)
            out = self.linear(max_out)
            out = nn.functional.dropout(out, p=0.5, training=self.training)
            return out
        pass
    pass


    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    for i in range(100):
        for epoch in range(1):
            model.train()
            loss_per_epoch = 0
            for batch in train_loader:
                optimizer.zero_grad()
                batch = batch.to(device)
                out = model(batch.x, batch.batch, batch.edge_index)
                out_mixup, batch.y_mixup = mixup_data(out, batch.y)
                log_prob = torch.nn.functional.log_softmax(out_mixup, dim=1)
                loss = -torch.sum(log_prob * batch.y_mixup) / 32
                loss_per_epoch += loss.item()
                loss.backward()
                optimizer.step()
            print('Loss per epoch: {}'.format(str(loss_per_epoch)))

    model.eval()
    correct = 0
    total = 0
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.batch, batch.edge_index)
        pred = torch.argmax(out, dim=1)
        batch.y = torch.argmax(batch.y, dim=1)
        total += 1
        if pred.item() == batch.y.item():
            correct += 1
    print('Eval Acc: {}'.format(str(correct / total)))


if __name__ == "__main__":
    main()

