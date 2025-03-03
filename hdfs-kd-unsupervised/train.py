import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Parameter
from torch.nn.modules.module import Module
from tqdm import tqdm
import math
import csv
from time import time 
from torchinfo import summary
from utils import save_model, model_train, read_train_data, load_train_data, read_data, load_data
from utils import DistilLog
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 2
batch_size = 50
learning_rate = 0.005
hidden_size = 32
input_size = 30
sequence_length = 50
num_layers = 2

save_teacher_path = '../datasets/HDFS/model/unsup_teacher.pth'
save_noKD_path = '../datasets/HDFS/model/unsup_noKD.pth'
train_path = "../datasets/HDFS/unsup_train.csv"
Teacher = DistilLog(input_size, hidden_size, num_layers, is_bidirectional=False).to(device)
#noKD = DistilLog(input_size = input_size, hidden_size = 4, num_layers = 1, is_bidirectional=False).to(device)
summary(Teacher, input_size=(50, 50, 30))
#summary(noKD, input_size=(50, 50, 30))

#train_x = read_data(train_path, input_size, sequence_length)
#train_loader = load_data(train_x, batch_size)

#train_x = read_train_data(input_size, sequence_length)
#train_loader = load_train_data(train_x, batch_size)


Teacher = model_train(Teacher, learning_rate, num_epochs = 50)
#noKD = model_train(noKD, learning_rate, num_epochs = 5)
save_model(Teacher, save_teacher_path)
#save_model(noKD, save_noKD_path)