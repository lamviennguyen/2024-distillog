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
from tqdm import tqdm, trange
import torch.nn.functional as F
import math

from utils import DistilLog, load_model, save_model, mod

num_classes = 2
num_epochs = 100
batch_size = 50
input_size = 30
sequence_length = 50
num_layers = 1
hidden_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_path = ('../datasets/HDFS/unsup_train.csv')
save_teacher_path = ('../datasets/HDFS/model/unsup_teacher.pth')
save_student_path = ('../datasets/HDFS/model/unsup_student.pth')

logs_series = pd.read_csv('../datasets/HDFS/unsup_train.csv').values[:,0]

train_total = len(logs_series)
split = 20
sub = int(train_total/split)

vec = pd.read_csv('../datasets/HDFS/pca_vector.csv', header = None)
vec = np.array(vec)



def read_train_data(input_size, sequence_length, i):
    if i!=split-1:
        logs_data = logs_series[i*sub:(i+1)*sub]
    else:
        logs_data = logs_series[i*sub:]
    logs = []

    for j in range(1, len(logs_data)):
        ori_seq = [
            int(eventid) for eventid in logs_data[j].split()]
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
    train_x = np.reshape(train_x, (train_x.shape[0], -1, input_size))

    return train_x

def load_train_data(data, batch_size):
    tensor_data = torch.Tensor(data)
    data_loader = DataLoader(tensor_data, batch_size=batch_size)
    return data_loader



def train_step(
    Teacher,
    Student,
    optimizer,
    divergence_loss_fn,
    temp,
    alpha,
    epoch,
    device
):
    losses = []
    for idx in trange(0, split):
        train_x = read_train_data(input_size = 30, sequence_length = 50, i = idx)
        train_loader = load_train_data(train_x, batch_size = 50) 
        for batch_idx, data in enumerate(train_loader):
            # Get data to cuda if possible
            data = data.to(device)

            # forward
            with torch.no_grad():
                teacher_preds = Teacher(data)['y_pred']

            return_dict = Student(data)
            student_preds = return_dict['y_pred']


            #print("Teacher:",teacher_preds)
            #print("Student:",student_preds)
            student_loss = return_dict['loss']
            
            ditillation_loss = divergence_loss_fn(
                F.softmax(student_preds / temp, dim=1),
                F.softmax(teacher_preds / temp, dim=1)
            )
            loss = alpha * student_loss + (1 - alpha) * ditillation_loss
            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
    
    avg_loss = sum(losses) / len(losses)
    return avg_loss

def teach(epochs, Teacher, Student, temp=7, alpha=0.3):
  Teacher = Teacher.to(device)
  Student = Student.to(device)
  student_loss_fn = nn.CrossEntropyLoss()
  divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")
  optimizer = torch.optim.Adam(Student.parameters(), lr=0.01)

  Teacher.eval()
  Student.train()
  for epoch in range(epochs):
      loss = train_step(
          Teacher,
          Student,
          optimizer,
          divergence_loss_fn,
          temp,
          alpha,
          epoch,
          device
      )

      print(f"Epoch {epoch+1} -  Loss:{loss:.2f}")

Teacher = DistilLog(input_size = input_size, hidden_size=32, num_layers = 2, is_bidirectional=False).to(device)
Student = DistilLog(input_size = input_size, hidden_size=4, num_layers = 2, is_bidirectional=False).to(device)

Teacher = load_model(Teacher, save_teacher_path)
teach(epochs=5, Teacher=Teacher, Student=Student, temp=7, alpha=0.3)
save_model(Student, save_student_path)
