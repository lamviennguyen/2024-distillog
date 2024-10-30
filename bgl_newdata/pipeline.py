import pickle
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Parameter
from torch.nn.modules.module import Module
from tqdm import tqdm
from time import time 
from utils import save_model, load_model
from utils import DistilLog
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


folderpath = '..\\newdatasets'
file_names = [f"BGL_20l_train_{x}.pkl" for x in [0.1, 0.2, 0.4, 0.6]]
num_classes = 2
batch_size = 50
learning_rate = 0.0003
hidden_size = 128
input_size = 300
sequence_length = 20
num_layers = 2


save_teacher_path = '../newdatasets/bgl_20_teacher.pth'
save_noKD_path = '../newdatasets/bgl_20_noKD.pth'

Teacher = DistilLog(input_size, hidden_size, num_layers, num_classes, is_bidirectional=False).to(device)


def train(model, train_loader, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    model.train()

    total_loss = 0
    for batch_idx, (data, target) in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        target = target.long()
        loss = criterion(output, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return model

def load_data(train_x, train_y, batch_size):
    train_y = train_y.astype(int)
    tensor_x = torch.Tensor(train_x)
    tensor_y = torch.from_numpy(train_y)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    return train_loader

def process_data_in_chunks(data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        log_vec_chunk = [np.array(item['Embeddings']) for item in chunk]
        label = [np.array(item['Label']) for item in chunk]
        yield log_vec_chunk, label

def start_train(file_path, model):
    with open(file_path, 'rb') as file:
        raw_data = pickle.load(file)
        full_data = raw_data[0] + raw_data[1]
        full_data = full_data[:-1]
    
    for i in range (1):
        print("epoch: ", i+1)    
        for log, label in process_data_in_chunks(full_data):
            log = np.array(log)
            label = np.array(label)
            data_loader = load_data(log, label, batch_size)
            model = train(model, data_loader, learning_rate)
    return model

for file_name in file_names: 
    file_path = os.path.join(folderpath, file_name) 
    if os.path.exists(file_path):
        Teacher = start_train(file_path, Teacher)
    
save_model(Teacher, save_teacher_path)


Teacher = load_model(Teacher, save_teacher_path)
#### test phase
with open('../newdatasets/BGL_20l_train_0.8.pkl', 'rb') as file:
    raw_test_data = pickle.load(file)

test_data = raw_test_data[0] + raw_test_data[1]
test_data = test_data[:-1]
def test(model, criterion = nn.CrossEntropyLoss()):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        TP = 0 
        FP = 0
        FN = 0 
        TN = 0

        for log, label in process_data_in_chunks(test_data):
            log = np.array(log)
            label = np.array(label)
            test_loader = load_data(log, label, batch_size)           
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                target = target.long()
                test_loss += criterion(output, target) # sum up batch loss
                
                output = torch.sigmoid(output)[:, 0].cpu().detach().numpy()
                predicted = (output < 0.2).astype(int)
                target = np.array([y.cpu() for y in target])

                TP += ((predicted == 1) * (target == 1)).sum()
                FP += ((predicted == 1) * (target == 0)).sum()
                FN += ((predicted == 0) * (target == 1)).sum()
                TN += ((predicted == 0) * (target == 0)).sum()
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)   
        accuracy = 100 * (TP + TN)/(TP + TN + FP + FN)
        #MCC = 100*(TP*TN + FP*FN)/math.sqrt((TP+FP)*(TN+FN)*(TN+FP)*(TP+FN))         
    return accuracy, test_loss, P, R, F1, TP, FP, TN, FN

accuracy, test_loss, P, R, F1, TP, FP, TN, FN = test(Teacher, criterion = nn.CrossEntropyLoss())

print('Result of testing teacher model')
print('false positive (FP): {}, false negative (FN): {}, true positive (TP): {}, true negative (TN): {}'.format(FP, FN, TP, TN))
print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%).')
print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%' .format(P, R, F1))



