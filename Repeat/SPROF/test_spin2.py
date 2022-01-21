#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_spin2.py
@Time    :   2022/01/20 11:16:18
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
import pandas as pd
from glob import glob

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
parser.add_argument('--run_fold', type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
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
    
######################################## function area ########################################

def test(dataset_name, args, device):
    ######################################## dataset setting area ########################################
    
    # load test dataset
    with open(f'{args.data_path}/spin2/{dataset_name}.pickle', 'rb') as f:
        names_list, sequences_dict, graphs_dict, labels_dict = pickle.load(f)
    # construct data loader
    test_loader, (args.node_features_dim, _) = get_loader(names_list, sequences_dict, graphs_dict, labels_dict,
                                                          batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    ######################################## model setting area ########################################
    
    # model
    model = DenseGCN(in_features=args.node_features_dim, hidden_features=args.hidden_features_dim, output_features=args.output_features_dim).to(device)
    
    ######################################## test area ########################################
    
    with open(f'{args.result_path}/spin2_seed_{args.seed}/{model.__class__.__name__}_{dataset_name}.txt', 'w') as f:
        
        if args.run_fold > 0:
            model_paths = glob(f'{args.result_path}/spin2_seed_{args.seed}/{model.__class__.__name__}_fold_{args.run_fold}.ckpt')
        else:
            model_paths = glob(f'{args.result_path}/spin2_seed_{args.seed}/{model.__class__.__name__}_fold_*.ckpt')
        
        message = f'{dataset_name} size: {len(names_list)}'
        f.write(message + '\n')
        print(message)
        all_predictions_list = list()

        # prediction
        for model_path in sorted(model_paths):
            fold_num = model_path[-6] if model_path[-6] != '0' else '10'
            
            # load weights
            model.eval()
            with torch.no_grad():
                model.load_state_dict(torch.load(model_path, map_location=device))
                results_dict, predictions_dict = loop(test_loader, model, optimizer=None, scheduler=None, device=device)
                all_predictions_list.append((fold_num, predictions_dict))
                
            # result summary
            message = f'fold {fold_num}: '
            for k, v in sorted(results_dict.items()):
                message += f'{k}: {v:.4f}   '
            f.write(message + '\n')
            print(message)
        
        # generate submission files
        data_dict = dict()
        data_dict['name'] = [name for name in names_list]
        data_dict['sequence'] = [sequences_dict[name] for name in names_list]
        data_dict['label'] = ensemble_true = [labels_dict[name] for name in names_list]
        for fold_num, predictions_dict in all_predictions_list:
            # [all_folds, len(sequence), 21]
            data_dict[f'fold_{fold_num}'] = [np.argmax(predictions_dict[name], axis=-1).tolist() for name in names_list]
        data_dict['average'] = ensemble_pred = [np.argmax(np.mean([predictions_dict[name] for fold_num, predictions_dict in all_predictions_list], axis=0), axis=-1).tolist() for name in names_list]
        data_frame = pd.DataFrame(data_dict, index=None)
        data_frame.to_csv(f'{args.result_path}/spin2_seed_{args.seed}/{model.__class__.__name__}_{dataset_name}.csv', sep=',', index=None)
        
        # ensemble result summary
        ensemble_results_dict = cal_metric(ensemble_true, ensemble_pred)
        message = f'ensemble: '
        for k, v in sorted(ensemble_results_dict.items()):
            message += f'{k}: {v:.4f}   '
        f.write(message + '\n')
        print(message + '\n')        
      
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
        
    ######################################## test area ########################################
    
    for dataset_name in ['test', 'casp']:
        test(dataset_name, args, device)
    