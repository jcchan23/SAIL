#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/01/24 11:12:56
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
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from torch.optim.lr_scheduler import _LRScheduler

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
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)


def loop(data_loader, model, optimizer, scheduler, device):
    batch_size = data_loader.batch_size
    data_loader = tqdm(data_loader) if optimizer is not None else data_loader
    
    loss_sum, y_true, y_pred = 0.0, list(), list()
    
    predictions = dict()
    
    for batch in data_loader:
        
        names, sequences, graphs, labels, masks = batch
        
        graphs = graphs.to(device)
        labels = labels.to(device)
        outputs = model(graphs, masks, device)
        
        # loss calculation
        # pad_sequence need cpu in model forward and need gpu in loss calculation
        masks = masks.to(device)
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
    auroc = metrics.roc_auc_score(binary_true, concatenate_pred)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)
    
    TN, FP, FN, TP = metrics.confusion_matrix(binary_true, binary_pred).ravel()
    sensitive = TP / (TP + FN)
    specificity = TN / (FP + TN)
    precision = TP / (TP + FP)
    
    return {'accuracy': accuracy, 'auroc': auroc, 'mcc': mcc, 'sensitive': sensitive, 'specificity': specificity, 'precision': precision,'threshold': best_threshold}


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.
    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch, init_lr, max_lr, final_lr):
        """
        Initializes the learning rate scheduler.
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after warmup_epochs).
        :param final_lr: The final learning rate (achieved after total_epochs).
        """
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.
        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]