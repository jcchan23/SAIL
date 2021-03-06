#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run_node.py
@Time    :   2022/03/08 14:35:24
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
import pickle
import warnings
from sklearn.model_selection import train_test_split

# private library
from dataset_node import get_loader
from utils_node import *

# model library
from model_node import CoMPT

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
parser.add_argument('--max_epochs', type=int, default=100,
                    help='The maximum epoch of training')
parser.add_argument('--batch_size', type=int, default=32,
                    help='The batch size')
parser.add_argument('--result_path', type=str, default='./result',
                    help='The name of result path, for logs, predictions, best models, etc.')
parser.add_argument('--run_fold', type=int, default=0,
                    help='The parallel running fold')
# dataset setting
parser.add_argument('--data_path', type=str, default='./data/preprocess',
                    help='The full path of features of the data.')
parser.add_argument('--data_name', type=str, default='1H',
                    help='the dataset name')
parser.add_argument('--resplit', action='store_true', default=False,
                    help="resplit the dataset with different comments")
# model setting
parser.add_argument('--hidden_features_dim', type=int, default=256,
                    help='the hidden features dimension')
parser.add_argument('--num_MHSA_layers', type=int, default=6,
                    help="the number of encoder layers")
parser.add_argument('--num_attention_heads', type=int, default=4,
                    help="the number of attention heads")
parser.add_argument('--num_FFN_layers', type=int, default=2,
                    help="the number of FFN layers")
parser.add_argument('--num_Generator_layers', type=int, default=2,
                    help="the number of generator layers")
parser.add_argument('--dropout', type=int, default=0.1,
                    help="the dropout rate")
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
    with open(f'{args.data_path}/{args.data_name}_train.pickle', 'rb') as f:
        # names = list(), others = dict() with key in names
        train_valid_smiles, train_valid_mols, train_valid_graphs, train_valid_labels = pickle.load(f)
    
    with open(f'{args.data_path}/{args.data_name}_test.pickle', 'rb') as f:
        test_smiles, test_mols, test_graphs, test_labels = pickle.load(f)
    
    # split dataset(if need!)
    if not os.path.exists(f'{args.result_path}/{args.data_name}_seed_{args.seed}') or args.resplit:
        
        # build a folder
        if not os.path.exists(f'{args.result_path}/{args.data_name}_seed_{args.seed}'):
            os.makedirs(f'{args.result_path}/{args.data_name}_seed_{args.seed}')

        # split 5% dataset as validation
        train_smiles, valid_smiles = train_test_split(train_valid_smiles, test_size=0.05, random_state=args.seed)
        
        with open(f'{args.result_path}/{args.data_name}_seed_{args.seed}/train.pickle','wb') as fw:
            pickle.dump(train_smiles, fw)
        with open(f'{args.result_path}/{args.data_name}_seed_{args.seed}/valid.pickle','wb') as fw:
            pickle.dump(valid_smiles, fw)
        with open(f'{args.result_path}/{args.data_name}_seed_{args.seed}/test.pickle','wb') as fw:
            pickle.dump(test_smiles, fw)
    
    # load pickle
    with open(f'{args.result_path}/{args.data_name}_seed_{args.seed}/train.pickle', 'rb') as f:
        train_smiles = pickle.load(f)
    with open(f'{args.result_path}/{args.data_name}_seed_{args.seed}/valid.pickle', 'rb') as f:
        valid_smiles = pickle.load(f)
    with open(f'{args.result_path}/{args.data_name}_seed_{args.seed}/test.pickle', 'rb') as f:
        test_smiles = pickle.load(f)
        
    # task setting
    args.task_type, args.task_metric, args.task_number = 'regression', 'mae', 1
    
    # construct data loader
    train_loader = get_loader(train_smiles, train_valid_mols, train_valid_graphs, train_valid_labels, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = get_loader(valid_smiles, train_valid_mols, train_valid_graphs, train_valid_labels, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = get_loader(test_smiles, test_mols, test_graphs, test_labels, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f'train size: {len(train_smiles):,} | valid size: {len(valid_smiles):,} | test size: {len(test_smiles):,}')
    
    ######################################## model setting area ########################################
    
    # model
    model = CoMPT(hidden_features=args.hidden_features_dim, output_features=args.task_number,
                  num_MHSA_layers=args.num_MHSA_layers, num_attention_heads=args.num_attention_heads,
                  num_FFN_layers=args.num_FFN_layers, num_Generator_layers=args.num_Generator_layers, dropout=args.dropout).to(device)
    
    # optimizer, scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = NoamLR(optimizer=optimizer, warmup_epochs=[5], total_epochs=[args.max_epochs],steps_per_epoch=len(train_loader),
                       init_lr=[0], max_lr=[args.learning_rate * 5], final_lr=[args.learning_rate])

    # initital weight and print model summary
    initialize_weights(model)
    print(model)
    
    ######################################## training area ########################################
    
    with open(f'{args.result_path}/{args.data_name}_seed_{args.seed}/{model.__class__.__name__}.txt', 'w') as f:
        best_epoch = -1
        best_metric = float('inf')
        best_results = None
        
        for epoch in range(1, args.max_epochs + 1):
            
            # train stage
            model.train()
            train_results = loop(data_loader=train_loader, model=model, optimizer=optimizer, scheduler=scheduler, device=device)
            
            # valid stage
            model.eval()
            with torch.no_grad():
                valid_results = loop(data_loader=valid_loader, model=model, optimizer=None, scheduler=None, device=device)
            
            # store model
            if args.task_type == 'regression':
                if valid_results[f'{args.task_metric}'] < best_metric:
                    best_epoch, best_metric, best_results = epoch, valid_results[f'{args.task_metric}'], valid_results
                    torch.save(model.state_dict(), f'{args.result_path}/{args.data_name}_seed_{args.seed}/{model.__class__.__name__}.ckpt')
            else:
                if valid_results[f'{args.task_metric}'] > best_metric:
                    best_epoch, best_metric, best_results = epoch, valid_results[f'{args.task_metric}'], valid_results
                    torch.save(model.state_dict(), f'{args.result_path}/{args.data_name}_seed_{args.seed}/{model.__class__.__name__}.ckpt')

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
        model = CoMPT(hidden_features=args.hidden_features_dim, output_features=args.task_number,
                        num_MHSA_layers=args.num_MHSA_layers, num_attention_heads=args.num_attention_heads,
                        num_FFN_layers=args.num_FFN_layers, num_Generator_layers=args.num_Generator_layers, dropout=args.dropout).to(device)
        model.eval()
        with torch.no_grad():
            model.load_state_dict(torch.load(f'{args.result_path}/{args.data_name}_seed_{args.seed}/{model.__class__.__name__}.ckpt', map_location=device))
            test_results = loop(data_loader=test_loader, model=model, optimizer=None, scheduler=None, device=device)

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
    