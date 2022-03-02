#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2022/01/24 11:12:06
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
from sklearn.model_selection import KFold

# private library
from dataset import get_loader
from utils import *

# model library
from model import MTDSite

######################################## parser area ########################################

parser = argparse.ArgumentParser()
# running setting
parser.add_argument('--seed', type=int, default=2021,
                    help="random seed")
parser.add_argument('--gpu', type=int, default=None,
                    help="set gpu")
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='The learning rate of ADAM optimization.')
parser.add_argument('--max_epochs', type=int, default=50,
                    help='The maximum epoch of training')
parser.add_argument('--batch_size', type=int, default=2,
                    help='The batch size')
parser.add_argument('--result_path', type=str, default='./result',
                    help='The name of result path, for logs, predictions, best models, etc.')
parser.add_argument('--run_fold', type=int, default=0, choices=[i for i in range(1, 11)],
                    help='The parallel running fold')
# dataset setting
parser.add_argument('--data_path', type=str, default='./data/preprocess',
                    help='The full path of features of the data.')
parser.add_argument('--data_name', type=str, choices=['carbohydrate', 'dna', 'peptide', 'rna', 'mix'],
                    help='The dataset name')
# model setting
parser.add_argument('--hidden_features_dim', type=int, default=128,
                    help="the hidden feature dimension in the network.")
parser.add_argument('--output_features_dim', type=int, default=2,
                    help="the output feature dimension in the network.")
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
    if args.data_name != 'mix':
        with open(f'{args.data_path}/{args.data_name}/train.pickle', 'rb') as f:
            # names = list(), others = dict() with key in names
            train_valid_names, train_valid_sequences, train_valid_graphs, train_valid_labels = pickle.load(f)
    else:
        train_valid_names, train_valid_sequences, train_valid_graphs, train_valid_labels = list(), dict(), dict(), dict()
        for dataset_name in ['carbohydrate', 'dna', 'peptide', 'rna']:
            with open(f'{args.data_path}/{dataset_name}/train.pickle', 'rb') as f:
                names_list, sequences_dict, graphs_dict, labels_dict = pickle.load(f)
                train_valid_names += names_list
                train_valid_sequences.update(sequences_dict)
                train_valid_graphs.update(graphs_dict)
                train_valid_labels.update(labels_dict)
            
    # split dataset (if need!)
    if not os.path.exists(f'{args.result_path}/{args.data_name}_seed_{args.seed}'):

        # build a folder
        os.makedirs(f'{args.result_path}/{args.data_name}_seed_{args.seed}')
        
        # train and valid split
        kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)
        for idx, (train_index, valid_index) in enumerate(kf.split(train_valid_names)):
            train_names, valid_names = np.array(train_valid_names)[train_index].tolist(), np.array(train_valid_names)[valid_index].tolist()
            with open(f'{args.result_path}/{args.data_name}_seed_{args.seed}/train_fold_{idx + 1}.txt', 'w') as fw:
                for name in train_names:
                    fw.write(name + '\n')
            with open(f'{args.result_path}/{args.data_name}_seed_{args.seed}/valid_fold_{idx + 1}.txt', 'w') as fw:
                for name in valid_names:
                    fw.write(name + '\n')
    
    # load train and valid dataset
    assert args.run_fold > 0
    with open(f'{args.result_path}/{args.data_name}_seed_{args.seed}/train_fold_{args.run_fold}.txt', 'r') as f:
        train_names = [line.strip() for line in f.readlines()]
    with open(f'{args.result_path}/{args.data_name}_seed_{args.seed}/valid_fold_{args.run_fold}.txt', 'r') as f:
        valid_names = [line.strip() for line in f.readlines()]
        
    # construct data loader
    train_loader, (args.node_features_dim, _) = get_loader(train_names, train_valid_sequences, train_valid_graphs, train_valid_labels,
                                                           batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader, (_, _) = get_loader(valid_names, train_valid_sequences, train_valid_graphs, train_valid_labels,
                                      batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # summary
    print(f'train size: {len(train_names):,} | valid size: {len(valid_names):,}')

    ######################################## model setting area ########################################
    
    # model
    model = MTDSite(args.node_features_dim, args.hidden_features_dim, args.output_features_dim).to(device)
    
    # optimizer, scheduler
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = NoamLR(optimizer, warmup_epochs=[5], total_epochs=[args.max_epochs], steps_per_epoch=len(train_loader),
                       init_lr=[0], max_lr=[args.learning_rate * 10], final_lr=[args.learning_rate])
    
    # initital weight and model summary
    initialize_weights(model)
    print(model)
    
    ######################################## training area ########################################
    
    # log
    with open(f'{args.result_path}/{args.data_name}_seed_{args.seed}/{model.__class__.__name__}_fold_{args.run_fold}.txt', 'w') as f:
        best_epoch, best_metric, best_results = -1, float('-inf'), None
        
        # epoch
        for epoch in range(1, args.max_epochs + 1):
            
            # train stage
            model.train()
            train_results, _ = loop(train_loader, model, optimizer, scheduler, device)
            
            # valid stage
            model.eval()
            with torch.no_grad(): 
                valid_results, _ = loop(valid_loader, model, optimizer=None, scheduler=None, device=device)
            
            # store model
            if valid_results['auroc'] > best_metric:
                best_epoch, best_metric, best_results = epoch, valid_results['auroc'], valid_results
                torch.save(model.state_dict(), f'{args.result_path}/{args.data_name}_seed_{args.seed}/{model.__class__.__name__}_fold_{args.run_fold}.ckpt')
            
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
        
        # total summary
        message = f'best epoch: {best_epoch}'
        message += '\nbest valid: '
        for k, v in sorted(best_results.items()):
            message += f'{k}: {v:.4f}   '
        message += '\n'
        f.write(message)
        print(message)