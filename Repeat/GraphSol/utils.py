#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/01/17 16:00:31
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import os
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn import metrics
from scipy.stats import pearsonr

######################################## function area ########################################

def seed_everything(seed=2021):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
        
        # names, sequences, graphs, labels, masks
        names, sequences, graphs, labels = batch
        
        graphs = graphs.to(device)
        # [batch_size, 1]
        labels = labels.to(device)
        # [batch_size, 1]
        outputs = model(graphs, device)
        
        # loss calculation
        loss = cal_loss(labels, outputs, task_type='regression')
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
        labels = labels.detach().cpu().numpy().tolist()
        outputs = outputs.detach().cpu().numpy().tolist()
        y_true.extend(labels)
        y_pred.extend(outputs)
        for name, output in zip(names, outputs):
            predictions[name] = output
        
        # clear cuda cache
        torch.cuda.empty_cache()

    # metric with threshold 0.5
    results = cal_metric(y_true, y_pred, threshold=0.5)
    if optimizer is not None:
        results['loss'] = loss_sum / (len(data_loader) * batch_size)
    return results, predictions


def cal_loss(y_true, y_pred, task_type='regression'):
    # y_true.shape = [batch_size, 1], y_pred.shape = [batch_size, 1]
    if task_type == 'regression':
        y_true = y_true.float()
        loss = F.mse_loss(y_pred, y_true, reduction='sum') / y_true.shape[1]
    elif task_type == 'classification':
        y_true = y_true.long()
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='sum') / y_true.shape[1]
    else:
        print("Not supported loss function!")
        assert False
    return loss


def cal_metric(y_true, y_pred, threshold=None):
    # y_true, y_pred.shape = numpy shape with (batch, 1)
    threshold = 0.5 if threshold is None else threshold
    concatenate_true, concatenate_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    
    binary_true = [1 if true >= threshold else 0 for true in concatenate_true]
    binary_pred = [1 if pred >= threshold else 0 for pred in concatenate_pred]
    
    rmse = np.sqrt(metrics.mean_squared_error(concatenate_true, concatenate_pred))
    pearson = pearsonr(concatenate_true, concatenate_pred)[0]
    r2 = metrics.r2_score(concatenate_true, concatenate_pred)
    
    accuracy = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    auroc = metrics.roc_auc_score(binary_true, concatenate_pred)
    
    return {'rmse': rmse, 'pearson': pearson, 'r2': r2,
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auroc': auroc}
    