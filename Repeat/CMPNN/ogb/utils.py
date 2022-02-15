#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2022/02/15 14:58:33
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)


def loop(data_loader, model, optimizer, scheduler, loss_name, evaluator, data_mean, data_std, device):
    batch_size = data_loader.batch_size
    data_loader = tqdm(data_loader) if optimizer is not None else data_loader
    
    loss_sum, y_true, y_pred = 0.0, list(), list()
    for batch in data_loader:
        graphs, labels = batch
        graphs = graphs.to(device)
        labels = labels.to(device)
        
        # (batch_size, args.task_number)
        output = model(graphs, device)
        
        # loss calculation
        loss = cal_loss(y_true=labels, y_pred=output, loss_name=loss_name, data_mean=data_mean, data_std=data_std, device=device)
        loss_sum += loss.data

        if optimizer is not None:
            # clear gradients for this training step
            optimizer.zero_grad()
            # back propagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()
        
        # NormLR need step every batch
        if scheduler is not None:
            scheduler.step()
        
        # collect result
        y_true += labels.detach().cpu().numpy().tolist()
        y_pred += output.detach().cpu().numpy().tolist()
        
        # clear cuda cache
        torch.cuda.empty_cache()
    
    # metric calculation
    loss_avg = loss_sum / (len(data_loader) * batch_size)
    data_mean, data_std = data_mean.detach().cpu().numpy(), data_std.detach().cpu().numpy()
    y_true, y_pred = np.array(y_true), np.array(y_pred) * data_std + data_mean
    evaluate_avg = evaluator.eval({'y_true':y_true, 'y_pred': y_pred})
    
    return dict({'loss': loss_avg}, **evaluate_avg)


def cal_loss(y_true, y_pred, loss_name, data_mean, data_std, device):
    # y_true, y_pred.shape = tensor shape with (batch, task_number), data_mean, data_std.shape = tensor shape with (1, task_number)
    y_true = torch.where(torch.isnan(y_true), torch.full_like(y_true, -1), y_true)
    y_true = torch.true_divide((y_true - data_mean), data_std)
    if loss_name == 'mse':
        # convert labels to float
        y_true = y_true.float()
        loss = F.mse_loss(y_pred, y_true, reduction="sum") / y_true.shape[1]
    elif loss_name == 'bce':
        # convert labels to long
        y_true = y_true.long()
        # find all -1 in y_true
        y_mask = torch.where(y_true == -1, torch.tensor([0]).to(device), torch.tensor([1]).to(device))
        y_cal_true = torch.where(y_true == -1, torch.tensor([0]).to(device), y_true)
        loss = F.binary_cross_entropy_with_logits(y_pred, y_cal_true.float(), reduction='none') * y_mask
        loss = loss.sum() / y_true.shape[1]
    else:
        raise "please refer loss function!"
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
