#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_spin2.py
@Time    :   2022/01/17 15:42:34
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import os
import torch
import pickle
import argparse
import warnings
import numpy as np
from sklearn.model_selection import KFold

# private library
from dataset import get_loader
from utils import *

# model library
from model import DenseGCN

######################################## parser area ########################################
parser = argparse.ArgumentParser()
# models setting
parser.add_argument('--seed', type=int, default=2021,
                    help="random seed")
parser.add_argument('--gpu', type=int, default=None,
                    help="set gpu")
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of workers')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='The learning rate of ADAM optimization.')
parser.add_argument('--max_epochs', type=int, default=100,
                    help='The maximum epoch of training')
parser.add_argument('--batch_size', type=int, default=8,
                    help='The batch size')
parser.add_argument('--result_path', type=str, default='./result',
                    help='The name of result path, for logs, predictions, best models, etc.')
parser.add_argument('--run_fold', type=int, default=0, choices=[1, 2, 3, 4, 5],
                    help='The parallel running fold')
# dataset setting
parser.add_argument('--data_path', type=str, default='./data/preprocess',
                    help='The full path of features of the data.')
# model setting
parser.add_argument('--hidden_features_dim', type=int, default=256,
                    help='the model hidden features dimension')
parser.add_argument('--output_features_dim', type=int, default=21,
                    help="the output dimension")
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
    
    # load total dataset
    with open(f'{args.data_path}/spin2/train.pickle', 'rb') as f:
        # names = list(), others = dict() with key in names
        train_valid_names, train_valid_sequences, train_valid_graphs, train_valid_labels = pickle.load(f)

    # cv split(if need!)
    if not os.path.exists(f'{args.result_path}/spin2_seed_{args.seed}'):
        
        # build a folder
        os.makedirs(f'{args.result_path}/spin2_seed_{args.seed}')
        
        # train and valid split
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        for idx, (train_index, valid_index) in enumerate(kf.split(train_valid_names)):
            train_names, valid_names = np.array(train_valid_names)[train_index].tolist(), np.array(train_valid_names)[valid_index].tolist()
            with open(f'{args.result_path}/spin2_seed_{args.seed}/train_fold_{idx + 1}.txt', 'w') as tw:
                for pdb in train_names:
                    tw.write(pdb + '\n')
            with open(f'{args.result_path}/spin2_seed_{args.seed}/valid_fold_{idx + 1}.txt', 'w') as vw:
                for pdb in valid_names:
                    vw.write(pdb + '\n')

    # load train and valid dataset
    assert args.run_fold > 0
    with open(f'{args.result_path}/spin2_seed_{args.seed}/train_fold_{args.run_fold}.txt', 'r') as f:
        train_names = [line.strip() for line in f.readlines()]
    with open(f'{args.result_path}/spin2_seed_{args.seed}/valid_fold_{args.run_fold}.txt', 'r') as f:
        valid_names = [line.strip() for line in f.readlines()]
        
    train_loader, (args.node_features_dim, args.edge_features_dim) = get_loader(train_names, train_valid_sequences, train_valid_graphs, train_valid_labels,
                                                                                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader, (_, _) = get_loader(valid_names, train_valid_sequences, train_valid_graphs, train_valid_labels,
                                      batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f'train size: {len(train_names):,} | valid size: {len(valid_names):,}')

    ######################################## model setting area ########################################

    # define a model
    model = DenseGCN(in_features=args.node_features_dim, hidden_features=args.hidden_features_dim, output_features=args.output_features_dim).to(device)

    # optimizer, scheduler, loss function
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = NoamLR(optimizer, warmup_epochs=[5], total_epochs=[args.max_epochs], steps_per_epoch=len(train_loader),
                       init_lr=[0], max_lr=[args.learning_rate * 10], final_lr=[args.learning_rate])

    # model summary
    print(model)

    ######################################## training area ########################################
    
    with open(f'{args.result_path}/spin2_seed_{args.seed}/{model.__class__.__name__}_fold_{args.run_fold}.txt', 'w') as f:
        best_epoch, best_metric, best_results = -1, float('-inf'), None
        for epoch in range(1, args.max_epochs + 1):
            # train stage
            model.train()
            train_results, _ = loop(data_loader=train_loader, model=model, optimizer=optimizer, scheduler=scheduler, device=device)

            # valid stage
            model.eval()
            with torch.no_grad():
                valid_results, _ = loop(data_loader=valid_loader, model=model, optimizer=None, scheduler=None, device=device)

            if valid_results['accuracy'] > best_metric:
                best_epoch = epoch
                best_metric = valid_results['accuracy']
                best_results = valid_results
                torch.save(model.state_dict(), f'{args.result_path}/spin2_seed_{args.seed}/{model.__class__.__name__}_fold_{args.run_fold}.ckpt')

            # epoch summary
            message = f'epoch: {epoch}'
            message += f'\ntrain: '
            for k, v in sorted(train_results.items()):
                message += f'{k}: {v:.4f}   '
            message += f'\nvalid: '
            for k, v in sorted(valid_results.items()):
                message += f'{k}: {v:.4f}   '
            message += '\n'
            f.write(message)
            print(message)
        
        # result summary
        message = f'best epoch: {best_epoch}'
        message += '\nbest valid: '
        for k, v in sorted(best_results.items()):
            message += f'{k}: {v:.4f}   '
        message += '\n'
        f.write(message)
        print(message)
