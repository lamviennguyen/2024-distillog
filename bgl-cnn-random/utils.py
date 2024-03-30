import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.modules.module import Module
from torchinfo import summary
from tqdm import tqdm
import csv
from time import time
from torch.nn import functional as F


device = torch.device('cuda')

def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    return model


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def mod(l, n):
    """ Truncate or pad a list """
    r = l[-1 * n:]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r


def read_data(path, input_size, sequence_length):
    fi = pd.read_csv('../datasets/BGL/pca_vector.csv', header = None)
    vec = []
    vec = fi
    vec = np.array(vec)

    logs_series = pd.read_csv(path)
    logs_series = logs_series.values
    label = logs_series[:, 1]
    logs_data = logs_series[:, 0]
    logs = []
    for i in range(0, len(logs_data)):
        ori_seq = [
            int(eventid) for eventid in logs_data[i].split()]
        seq_pattern = mod(ori_seq, sequence_length)
        vec_pattern = []

        for event in seq_pattern:
            if event == 0:
                vec_pattern.append([-1] * input_size)
            else:
                vec_pattern.append(vec[event - 1])
        logs.append(vec_pattern)
    logs = np.array(logs)
    train_x = logs
    train_y = np.array(label)
    train_x = np.reshape(train_x, (train_x.shape[0], -1, input_size))
    train_y = train_y.astype(int)

    return train_x, train_y


def load_data(train_x, train_y, batch_size):
    tensor_x = torch.Tensor(train_x)
    tensor_y = torch.from_numpy(train_y)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader


def train(model, train_loader, learning_rate, num_epochs):
    criterion = nn.CrossEntropyLoss()
    summary(model, input_size=(50, 50, 30))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    model.train()

    for epoch in range(num_epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        total_loss = 0
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()


            if (batch_idx + 1) % 10 == 0:
                done = (batch_idx + 1) * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(
                    f'Train Epoch: {epoch + 1}/{num_epochs} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {total_loss:.6f}')

    return model


if __name__ == '__main__':
    inp = torch.rand((1, 8, 30))
    print(inp.shape)
    out, _ = model(inp)
    print(out.shape)
