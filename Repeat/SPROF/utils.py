#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/01/17 15:43:56
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from sklearn import metrics
from tqdm import tqdm

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

        names, sequences, graphs, labels, masks = batch
        graphs = graphs.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        # (num_edge, 21)
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
        scoreMax = np.argmax(scores, axis=-1)
        # scoreMax = torch.max(scores, dim=1)[1]
        # scoreMax = scoreMax.detach().cpu().numpy()
        for name, (idx, length) in zip(names, masks):
            y_true.append(labels[idx:idx+length])
            y_pred.append(scoreMax[idx:idx+length])
            predictions[name] = scores[idx:idx+length, :].tolist()
        
        # clear cuda cache
        torch.cuda.empty_cache()

    # metric calculation
    if optimizer is not None:
        results = cal_metric(y_true, y_pred)
        results['loss'] = loss_sum / (len(data_loader) * batch_size)
    else:
        results = cal_metric(y_true, y_pred)
    return results, predictions


def cal_loss(y_true, y_pred, y_mask):
    total_loss = CELoss(label_smooth=0.1)(y_pred, y_true)
    # y_true.shape = [batch_num_nodes], y_pred.shape = [batch_num_nodes, 2], total_loss.shape = [batch_num_nodes]
    if y_mask is None:
        loss = torch.mean(total_loss)
    else:
        loss = 0.0
        for idx, length in y_mask:
            loss = loss + torch.mean(total_loss[idx:idx+length])
    return loss


def cal_metric(y_true, y_pred):
    acc_list = list()
    for true, pred in zip(y_true, y_pred):
        acc_list.append(metrics.accuracy_score(true, pred))
    return {'accuracy':np.nanmean(acc_list)}


class CELoss(nn.Module):
    """ Cross Entropy Loss with label smoothing """

    def __init__(self, label_smooth=None, class_num=21):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):  # pred.shape = (L, class_num)
        # target.shape = (L, class_num) with pre-designed label
        if len(target.shape) == 2:
            prob = F.log_softmax(pred, dim=1)
            loss = -1.0 * torch.sum(prob * target, dim=1)
        # target.shape = (L) with label smooth
        elif self.label_smooth is not None:
            prob = F.log_softmax(pred, dim=1)
            # (L) -> (L, class_num)
            target = F.one_hot(target, self.class_num)
            target = torch.clamp(target.float(), min=self.label_smooth / (self.class_num - 1), max=1.0 - self.label_smooth)
            # (L, class_num) -> (L)
            loss = -1.0 * torch.sum(prob * target, dim=1)
        # target.shape = (L) without label smooth
        else:
            loss = F.cross_entropy(pred, target, reduction='none')
        
        return loss


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
