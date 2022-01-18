#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/01/18 14:07:51
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import os
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn import metrics

######################################## function area ########################################

def seed_everything(seed=2021):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def load_dataset(path, mode=None):
    if mode == 'pickle':
        with open(path, 'rb') as f:
            names, sequences, labels = pickle.load(f)
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
        
        names, sequences, labels = list(), list(), list()    
        for idx, line in enumerate(lines):
            line = line.strip()
            if line == "":
                continue
            elif idx % 3 == 0:
                names.append(line[1:])
            elif idx % 3 == 1:
                sequences.append(line)
            else:
                labels.append([int(num) for num in line])
        
    return names, sequences, labels


def initialize_weights(model):
    """
    Initializes the weights of a model in place.

    :param model: An nn.Module.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
            

def loop(data_loader, model, optimizer, scheduler, device):
    batch_size = data_loader.batch_size
    data_loader = tqdm(data_loader) if optimizer is not None else data_loader
    
    loss_sum, y_true, y_pred = 0.0, list(), list()
    
    predictions = dict()
    
    for batch in data_loader:
        
        names, sequences, graphs, labels, masks = batch
        
        graphs = graphs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)    
        outputs = model(graphs, device)
        
        # loss calculation
        loss = cal_loss(labels, outputs, masks)
        loss_sum += loss.data
        
        if optimizer is not None:
            # clear gradients for this training step
            optimizer.zero_grad()
            # back propagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()
        
        # NormLR needs step every batch
        if scheduler is not None:
            scheduler.step()
        
        # collect result
        labels = labels.detach().cpu().numpy()
        scores = torch.softmax(outputs, dim=1)
        scores = scores.detach().cpu().numpy()
        scores = scores[:, 1]
        for name, (idx, length) in zip(names, masks):
            y_true.append(labels[idx:idx+length].tolist())
            y_pred.append(scores[idx:idx+length].tolist())
            predictions[name] = scores[idx:idx+length].tolist()
        
        # clear cuda cache
        torch.cuda.empty_cache()

    # train with threshold = 0.5, test without using threshold
    if optimizer is not None:
        results = cal_metric(y_true, y_pred, best_threshold=0.5)
        results['loss'] = loss_sum / (len(data_loader) * batch_size)
    else:
        results = cal_metric(y_true, y_pred, best_threshold=None)
    return results, predictions


def cal_loss(y_true, y_pred, y_mask):
    # y_true.shape = [batch_num_nodes], y_pred.shape = [batch_num_nodes, 2], total_loss.shape = [batch_num_nodes]
    total_loss = nn.CrossEntropyLoss(reduction='none')(y_pred, y_true)
    loss = 0.0
    for idx, length in y_mask:
        loss = loss + torch.mean(total_loss[idx:idx+length])
    return loss


def cal_metric(y_true, y_pred, best_threshold=None):
    concatenate_true, concatenate_pred  = np.concatenate(y_true, axis=-1), np.concatenate(y_pred, axis=-1)
    
    if best_threshold is None:
        best_f1, best_threshold = 0, 0
        for threshold in range(100):
            threshold /= 100
            binary_true = concatenate_true
            binary_pred = [1 if pred >= threshold else 0 for pred in concatenate_pred]
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold
                
    binary_true = concatenate_true
    binary_pred = [1 if pred >= best_threshold else 0 for pred in concatenate_pred]
    
    accuracy = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    auroc = metrics.roc_auc_score(binary_true, concatenate_pred)
    precisions, recalls, _ = metrics.precision_recall_curve(binary_true, concatenate_pred)
    auprc = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auroc': auroc, 'auprc': auprc, 'mcc': mcc, 'threshold': best_threshold}
