import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange, tqdm  

import math
import copy
from time import time 
from utils import load_data, load_model, DistilLog, save_model 



batch_size = 100
input_size = 30
sequence_length = 50
hidden_size = 128
num_layers = 2
num_classes = 2 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
save_teacher_path = '../datasets/HDFS/model/unsup_teacher.pth'
save_student_path = '../datasets/HDFS/model/unsup_student.pth'
save_noKD_path = '../datasets/HDFS/model/unsup_noKD.pth'
test_path = '../datasets/HDFS/unsup_test.csv'
#test_path = '../datasets/HDFS/train.csv'


fi = pd.read_csv('../datasets/HDFS/pca_vector.csv', header = None)
vec = []
vec = fi
vec = np.array(vec)

test_logs_series = pd.read_csv(test_path)
test_logs_series = test_logs_series.values
test_total = len(test_logs_series)
sub = batch_size
if test_total % sub == 0:
    split = int(test_total/sub)
else:
    split = int(test_total/sub) + 1

#anomaly_ratio = sum(test_logs_series[:,1])/test_total


def mod(l, n):
    """ Truncate or pad a list """
    r = l[-1*n:]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r

def load_test(i):
    if i!=split-1:
        label = test_logs_series[i*sub:(i+1)*sub,1]
        logs_data = test_logs_series[i*sub:(i+1)*sub,0]
    else:
        label = test_logs_series[i*sub:,1]
        logs_data = test_logs_series[i*sub:,0]
    logs = []

    for logid in range(0,len(logs_data)):
        ori_seq = [
            int(eventid) for eventid in logs_data[logid].split()]
        seq_pattern = mod(ori_seq, sequence_length)
        vec_pattern = []

        for event in seq_pattern:
            if event == 0:
                vec_pattern.append([-1]*input_size)
            else:
                vec_pattern.append(vec[event-1])  
        logs.append(vec_pattern)
    logs = np.array(logs)
    train_x = logs
    train_y = label
    train_x = np.reshape(train_x, (train_x.shape[0], -1, input_size))
    train_y = train_y.astype(int)
    return train_x, train_y

def test(model, anomaly_ratio):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        TP = 0 
        FP = 0
        FN = 0 
        TN = 0
        for i in trange (0, split):        #################################################
            test_x, test_y = load_test(i)
            test_loader = load_data(test_x, test_y, batch_size)            
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                y_pred = []

                return_dict = model(data)
                y_pred = return_dict["y_pred"]
                y_pred = y_pred[:, -1].data.cpu().numpy().reshape(-1)
                
                thre = np.percentile(y_pred, anomaly_ratio)

                predicted = (y_pred > thre).astype(int)
                #y = (window_anomalies > 0).astype(int)
                
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
    return accuracy, P, R, F1, TP, FP, TN, FN


def mad_score(points):
    m = np.median(points)
    ad = np.abs(points - m)
    mad = np.median(ad)
    
    return 0.6745 * ad / mad

def test_new(model, threshold):
    model.eval()
    with torch.no_grad():
        TP = 0 
        FP = 0
        FN = 0 
        TN = 0
        for i in trange (0, split):        #################################################
            test_x, test_y = load_test(i)
            test_loader = load_data(test_x, test_y, batch_size)            
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                return_dict = model(data)
                reconstruction = return_dict['reconstruction']

                l2 = data - reconstruction
                mse = np.power(l2.cpu(), 2).mean(dim = -1)
                #print(mse)
                #z_scores = mad_score(mse).mean().data.cpu().numpy()
                z_scores = mad_score(mse).data.cpu().numpy()

                thre = [threshold*100 for i in range(data.shape[0])]
                predicted = np.array([z.mean() for z in z_scores] > thre).astype(int)
                
                
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
    return accuracy, P, R, F1, TP, FP, TN, FN

def main():     
     
    print(device)
    '''
    teacher = DistilLog(input_size = input_size, hidden_size = 32, num_layers = num_layers, is_bidirectional=False).to(device)
    teacher = load_model(teacher, save_teacher_path)
    
    #noKD = DistilLog(input_size = input_size, hidden_size = 4, num_layers = 1, is_bidirectional=False).to(device)
    #noKD = load_model(noKD, save_noKD_path)

    
    #best_thre = sum(test_logs_series[:,1])/test_total #anomaly ratio
    best_thre = 1000
    max_F1 = 0.0
    threshold = 5

    
    for i in range(10):
        threshold += 1    

        accuracy, P, R, F1, TP, FP, TN, FN = test(teacher, anomaly_ratio = threshold)
        if F1 > max_F1:
            best_thre = threshold
            max_F1 = F1 
        print(i+1, '[TP, TN, FP, FN] =',TP, TN, FP, FN)
        print('thre: {:.3f}'.format(threshold))
        print('Accuracy: {:.3f}%, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%' .format(accuracy, P, R, F1))

    print(best_thre, max_F1)
    '''


    
    student = DistilLog(input_size = input_size, hidden_size = 4 , num_layers = 2, is_bidirectional=False).to(device)
    student = load_model(student, save_student_path)
    

    
    best_thre = 1000
    max_F1 = 0.0
    threshold = 15

    
    for i in range(20):
        threshold += 1    

        accuracy, P, R, F1, TP, FP, TN, FN = test(student, anomaly_ratio = threshold)
        if F1 > max_F1:
            best_thre = threshold
            max_F1 = F1 
        print(i+1, '[TP, TN, FP, FN] =',TP, TN, FP, FN)
        print('thre: {:.3f}'.format(threshold))
        print('Accuracy: {:.3f}%, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%' .format(accuracy, P, R, F1))

    print(best_thre, max_F1)

    

    """
    
    noKD = DistilLog(input_size = input_size, hidden_size = 4, num_layers = 1, is_bidirectional=False).to(device)
    noKD = load_model(noKD, save_noKD_path)
    print(noKD)
    best_thre = 1000
    max_F1 = 0.0
    threshold = 0

    
    for i in range(100):
        threshold += 1
        accuracy, P, R, F1, TP, FP, TN, FN = test(noKD, anomaly_ratio = threshold)
        if F1 > max_F1:
            best_thre = threshold
            max_F1 = F1 
        print(i+1, '[TP, TN, FP, FN] =',TP, TN, FP, FN)
        print('thre: {:.3f}%'.format(threshold))
        print('Accuracy: {:.3f}%, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%' .format(accuracy, P, R, F1))


    print(best_thre, max_F1)
    """
if __name__ == "__main__":

    main()

