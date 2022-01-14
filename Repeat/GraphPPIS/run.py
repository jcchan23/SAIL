#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run.py
@Time    :   2022/01/14 08:52:26
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
from model import GraphPPIS

######################################## parser area ########################################

parser = argparse.ArgumentParser()
# running setting
parser.add_argument('--seed', type=int, default=2021,
                    help="random seed")
parser.add_argument('--gpu', type=int, default=None,
                    help="set gpu")
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of workers')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='The learning rate of ADAM optimization.')
parser.add_argument('--max_epochs', type=int, default=50,
                    help='The maximum epoch of training')
parser.add_argument('--batch_size', type=int, default=2,
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
                    help="the hidden feature dimension in the network.")
parser.add_argument('--output_features_dim', type=int, default=2,
                    help="the output feature dimension in the network.")
parser.add_argument('--num_layers', type=int, default=8,
                    help="the number of convolution layers in the network.")
parser.add_argument('--dropout', type=int, default=0.1,
                    help="the dropout rate in the network.")
parser.add_argument('--lamda', type=int, default=1.5,
                    help="the lambda number in the network.")
parser.add_argument('--alpha', type=int, default=0.7,
                    help="the alpha number in the network.")
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
    with open(f'{args.data_path}/Train_335.pickle', 'rb') as f:
        # names = list(), others = dict() with key in names
        train_valid_names, train_valid_sequences, train_valid_graphs, train_valid_labels = pickle.load(f)

    # split dataset (if need!)
    if not os.path.exists(f'{args.result_path}/seed_{args.seed}'):

        # build a folder
        os.makedirs(f'{args.result_path}/seed_{args.seed}')
        
        # train and valid split
        kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
        for idx, (train_index, valid_index) in enumerate(kf.split(train_valid_names)):
            train_names, valid_names = np.array(train_valid_names)[train_index].tolist(), np.array(train_valid_names)[valid_index].tolist()
            with open(f'{args.result_path}/seed_{args.seed}/train_fold_{idx + 1}.txt', 'w') as fw:
                for name in train_names:
                    fw.write(name + '\n')
            with open(f'{args.result_path}/seed_{args.seed}/valid_fold_{idx + 1}.txt', 'w') as fw:
                for name in valid_names:
                    fw.write(name + '\n')
    
    # load train and valid dataset
    assert args.run_fold > 0
    with open(f'{args.result_path}/seed_{args.seed}/train_fold_{args.run_fold}.txt', 'r') as f:
        train_names = [line.strip() for line in f.readlines()]
    with open(f'{args.result_path}/seed_{args.seed}/valid_fold_{args.run_fold}.txt', 'r') as f:
        valid_names = [line.strip() for line in f.readlines()]
        
    # load test dataset, maybe use exec() to reduce the repeat code
    with open(f'{args.data_path}/Test_315.pickle', 'rb') as f:
        test_315_names, test_315_sequences, test_315_graphs, test_315_labels = pickle.load(f)
    with open(f'{args.data_path}/Test_60.pickle', 'rb') as f:
        test_60_names, test_60_sequences, test_60_graphs, test_60_labels = pickle.load(f)
    with open(f'{args.data_path}/Btest_31.pickle', 'rb') as f:
        btest_31_names, btest_31_sequences, btest_31_graphs, btest_31_labels = pickle.load(f)
    with open(f'{args.data_path}/UBtest_31.pickle', 'rb') as f:
        ubtest_31_names, ubtest_31_sequences, ubtest_31_graphs, ubtest_31_labels = pickle.load(f) 
    
    # construct data loader
    train_loader, (args.node_features_dim, _) = get_loader(train_names, train_valid_sequences, train_valid_graphs, train_valid_labels,
                                                           batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader, (_, _) = get_loader(valid_names, train_valid_sequences, train_valid_graphs, train_valid_labels,
                                      batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_315_loader, (_, _) = get_loader(test_315_names, test_315_sequences, test_315_graphs, test_315_labels,
                                         batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_60_loader, (_, _) = get_loader(test_60_names, test_60_sequences, test_60_graphs, test_60_labels,
                                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    btest_31_loader, (_, _) = get_loader(btest_31_names, btest_31_sequences, btest_31_graphs, btest_31_labels,
                                         batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    ubtest_31_loader, (_, _) = get_loader(ubtest_31_names, ubtest_31_sequences, ubtest_31_graphs, ubtest_31_labels,
                                          batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # summary
    print(f'train size: {len(train_names):,} | valid size: {len(valid_names):,} |',
          f'test_315 size: {len(test_315_names):,} | test_60 size: {len(test_60_names):,} |',
          f'btest_31_size: {len(btest_31_names):,} | ubtest_31 size: {len(ubtest_31_names):,}')

    ######################################## model setting area ########################################
    
    # model
    model = GraphPPIS(args.node_features_dim, args.hidden_features_dim, args.output_features_dim, 
                      args.num_layers, args.dropout, args.lamda, args.alpha).to(device)
    
    # optimizer, scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None
    
    # initital weight and model summary
    initialize_weights(model)
    print(model)
    
    ######################################## training area ########################################
    
    # log
    with open(f'{args.result_path}/seed_{args.seed}/{model.__class__.__name__}_fold_{args.run_fold}.txt', 'w') as f:
        best_epoch, best_metric, best_results = -1, float('-inf'), None
        
        for epoch in range(1, args.max_epochs + 1):
            
            # train stage
            model.train()
            train_results = loop(train_loader, model, optimizer, scheduler, device)
            
            # valid stage
            model.eval()
            with torch.no_grad(): 
                valid_results = loop(valid_loader, model, optimizer=None, scheduler=None, device=device)
            
            # store model
            if valid_results['auprc'] > best_metric:
                best_epoch, best_metric, best_results = epoch, valid_results['auprc'], valid_results
                torch.save(model.state_dict(), f'{args.result_path}/seed_{args.seed}/{model.__class__.__name__}_fold_{args.run_fold}.ckpt')
            
            # result summary
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
        
        ######################################## test area ########################################
        
        # test stage
        model = GraphPPIS(args.node_features_dim, args.hidden_features_dim, args.output_features_dim, 
                          args.num_layers, args.dropout, args.lamda, args.alpha).to(device)
        model.eval()
        with torch.no_grad():
            model.load_state_dict(torch.load(f'{args.result_path}/seed_{args.seed}/{model.__class__.__name__}_fold_{args.run_fold}.ckpt', map_location=device))
            test_315_results = loop(test_315_loader, model, optimizer=None, scheduler=None, device=device)
            test_60_results = loop(test_60_loader, model, optimizer=None, scheduler=None, device=device)
            btest_31_results = loop(btest_31_loader, model, optimizer=None, scheduler=None, device=device)
            ubtest_31_results = loop(ubtest_31_loader, model, optimizer=None, scheduler=None, device=device)
        
        # result summary
        message = f'best epoch: {best_epoch}'
        message += '\nbest valid: '
        for k, v in sorted(best_results.items()):
            message += f'{k}: {v:.4f}   '
        message += '\ntest_315: '
        for k, v in sorted(test_315_results.items()):
            message += f'{k}: {v:.4f}   '
        message += '\ntest_60: '
        for k, v in sorted(test_60_results.items()):
            message += f'{k}: {v:.4f}   '
        message += '\nbtest_31: '
        for k, v in sorted(btest_31_results.items()):
            message += f'{k}: {v:.4f}   '
        message += '\nubtest_31: '
        for k, v in sorted(ubtest_31_results.items()):
            message += f'{k}: {v:.4f}   '
        message += '\n'
        f.write(message)
        print(message)
    