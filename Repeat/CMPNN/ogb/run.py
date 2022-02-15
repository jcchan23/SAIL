#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run.py
@Time    :   2022/02/15 11:54:21
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import torch
import os
import argparse
import numpy as np
import pickle
import warnings

from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl, Evaluator
from torch.utils.data import DataLoader

# private library
from utils import *

# model library
from model import CMPNN

######################################## parser area ########################################

parser = argparse.ArgumentParser()
# running setting
parser.add_argument('--seed', type=int, default=2021,
                    help="random seed")
parser.add_argument('--gpu', type=int, default=None,
                    help="set gpu")
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of workers')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='The learning rate of ADAM optimization.')
parser.add_argument('--max_epochs', type=int, default=30,
                    help='The maximum epoch of training')
parser.add_argument('--batch_size', type=int, default=50,
                    help='The batch size')
parser.add_argument('--result_path', type=str, default='./result',
                    help='The name of result path, for logs, predictions, best models, etc.')
parser.add_argument('--run_fold', type=int, default=0,
                    help='The parallel running fold')
# dataset setting
parser.add_argument('--data_name', type=str, default='ogbg-moltox21',
                    help='the dataset name')
# model setting
parser.add_argument('--hidden_features_dim', type=int, default=300,
                    help='the hidden features dimension')
parser.add_argument('--num_step_message_passing', type=int, default=3,
                    help="the number of CMPNN layers")
# args executing
args = parser.parse_args()
for arg in vars(args):
    print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))

######################################## main area ########################################

if __name__ == '__main__':
    
    ######################################## running setting area ########################################
    
    # warning
    warnings.filterwarnings("ignore")
    # seed
    seed_everything(args.seed)
    # device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    # result folder
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    
    ######################################## dataset setting area ########################################
    
    dataset = DglGraphPropPredDataset(name=args.data_name, root=f'{args.result_path}/ogbg_seed_{args.seed}')
    
    # task setting
    if args.data_name in ['ogbg-molbace','ogbg-molbbbp', 'ogbg-molclintox', 'ogbg-molsider', 'ogbg-moltox21', 'ogbg-moltoxcast', 'ogbg-molhiv', 'ogbg-molchembl']:
        args.task_type, args.task_loss, args.task_metric = 'classification','bce', 'rocauc'
    elif args.data_name in ['ogbg-molesol', 'ogbg-molfreesolv', 'ogbg-mollipo']:
        args.task_type, args.task_loss, args.task_metric = 'regression','mse', 'rmse'
    elif args.data_name in ['ogbg-molmuv', 'ogbg-molpcba']:
        args.task_type, args.task_loss, args.task_metric = 'classification','bce', 'ap'
    elif args.data_name == 'ogbg-ppa':
        args.task_type, args.task_loss, args.task_metric = 'classification','bce', 'acc'
    elif args.data_name == 'ogbg-code2':
        args.task_type, args.task_loss, args.task_metric = 'classification','bce', 'F1'
    else:
        raise "Not supported task setting, please refer the correct data name!"
    args.task_number = max([len(label) if len(graph.edata['feat'].shape) > 1 else 0 for graph, label in dataset])
    
    # data loader
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, collate_fn=collate_dgl)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=True, collate_fn=collate_dgl)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=True, collate_fn=collate_dgl)
    
    # normalize label with the shape of (1, task_number)
    if args.task_type == 'regression':
        train_labels = np.concatenate([label.numpy() for _, label in dataset[split_idx["train"]]], axis=0).reshape(-1, 1)
        label_mean = torch.from_numpy(np.nanmean(train_labels, axis=0, keepdims=True)).float().to(device)
        label_std = torch.from_numpy(np.nanstd(train_labels, axis=0, keepdims=True)).float().to(device)
    else:
        label_mean = torch.from_numpy(np.array([[0 for _ in range(args.task_number)]])).long().to(device)
        label_std = torch.from_numpy(np.array([[1 for _ in range(args.task_number)]])).long().to(device)
    
    print(f'train size: {len(split_idx["train"]):,} | valid size: {len(split_idx["valid"]):,} | test size: {len(split_idx["test"]):,}')

    ######################################## model setting area ########################################
    
    # model
    model = CMPNN(hidden_features=args.hidden_features_dim, output_features=args.task_number, num_step_message_passing=args.num_step_message_passing).to(device)
    
    # optimizer, scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = NoamLR(optimizer=optimizer, warmup_epochs=[2], total_epochs=[args.max_epochs],steps_per_epoch=len(train_loader),
                       init_lr=[args.learning_rate], max_lr=[args.learning_rate * 10], final_lr=[args.learning_rate])
    evaluator = Evaluator(name=args.data_name)

    # initital weight and print model summary
    initialize_weights(model)
    print(model)
    
    ######################################## training area ########################################
    
    with open(f'{args.result_path}/ogbg_seed_{args.seed}/{args.data_name.replace("-","_")}/{model.__class__.__name__}_fold_{args.run_fold}.txt', 'w') as f:
        best_epoch = -1
        best_metric = float('inf') if args.task_type == 'regression' else float('-inf')
        best_results = None
        
        for epoch in range(1, args.max_epochs + 1):
            
            # train stage
            model.train()
            train_results = loop(data_loader=train_loader, model=model, optimizer=optimizer, scheduler=scheduler,
                                 loss_name=args.task_loss, evaluator=evaluator, data_mean=label_mean, data_std=label_std, device=device)
            
            # valid stage
            model.eval()
            with torch.no_grad():
                valid_results = loop(data_loader=valid_loader, model=model, optimizer=None, scheduler=None,
                                     loss_name=args.task_loss, evaluator=evaluator, data_mean=label_mean, data_std=label_std, device=device)
            
            # store model
            if args.task_type == 'regression':
                if valid_results[f'{args.task_metric}'] < best_metric:
                    best_epoch, best_metric, best_results = epoch, valid_results[f'{args.task_metric}'], valid_results
                    torch.save(model.state_dict(), f'{args.result_path}/ogbg_seed_{args.seed}/{args.data_name.replace("-","_")}/{model.__class__.__name__}_fold_{args.run_fold}.ckpt')
            else:
                if valid_results[f'{args.task_metric}'] > best_metric:
                    best_epoch, best_metric, best_results = epoch, valid_results[f'{args.task_metric}'], valid_results
                    torch.save(model.state_dict(), f'{args.result_path}/ogbg_seed_{args.seed}/{args.data_name.replace("-","_")}/{model.__class__.__name__}_fold_{args.run_fold}.ckpt')
            
            # result summary
            message = f"epoch: {epoch}"
            message += "\ntrain: "
            for k, v in sorted(train_results.items()):
                message += f"{k}: {v:.4f}   "
            message += "\nvalid: "
            for k, v in sorted(valid_results.items()):
                message += f"{k}: {v:.4f}   "
            message += "\n"
            f.write(message)
            print(message)
    
        ######################################## test area ########################################
        
        # test stage
        model = CMPNN(hidden_features=args.hidden_features_dim, output_features=args.task_number, num_step_message_passing=args.num_step_message_passing).to(device)
        model.eval()
        with torch.no_grad():
            model.load_state_dict(torch.load(f'{args.result_path}/ogbg_seed_{args.seed}/{args.data_name.replace("-","_")}/{model.__class__.__name__}_fold_{args.run_fold}.ckpt', map_location=device))
            test_results = loop(data_loader=test_loader, model=model, optimizer=None, scheduler=None,
                                    loss_name=args.task_loss, evaluator=evaluator, data_mean=label_mean, data_std=label_std, device=device)

        # result summary
        message = f"best epoch: {best_epoch}"
        message += "\nbest valid: "
        for k, v in sorted(best_results.items()):
            message += f"{k}: {v:.4f}   "
        message += f"\ntest: "
        for k, v in sorted(test_results.items()):
            message += f"{k}: {v:.4f}   "
        message += "\n"
        f.write(message)
        print(message)
        