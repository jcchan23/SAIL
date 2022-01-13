######################################## import area ########################################

# common library
import torch
import os
import argparse
import numpy as np
import pickle
import warnings
from sklearn.model_selection import KFold, train_test_split

# private library
from dataset import get_loader
from utils import *

# model library
from model import CrystalNet

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
parser.add_argument('--total_folds', type=int, default=9,
                    help='The total folds in cv')
parser.add_argument('--run_fold', type=int, default=0,
                    help='The parallel running fold')
# dataset setting
parser.add_argument('--data_path', type=str, default='./data/preprocess',
                    help='The full path of features of the data.')
parser.add_argument('--data_name', type=str, default='mp',
                    help='the dataset name')
parser.add_argument('--task_name', type=str, default='band_gap',
                    help='the task name')
parser.add_argument('--resplit', action='store_true', default=False,
                    help="resplit the dataset with different comments")
# model setting
parser.add_argument('--hidden_features_dim', type=int, default=300,
                    help='the hidden features dimension')
parser.add_argument('--num_step_message_passing', type=int, default=2,
                    help="the number of CrystalNet layers")
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
    print('Loading dataset...')
    with open(f'{args.data_path}/{args.data_name}.pickle', 'rb') as f:
        # names = list(), others = dict() with key in names
        names, crystals, graphs, labels = pickle.load(f)
    
    # split dataset(if need!)
    if not os.path.exists(f'{args.result_path}/{args.data_name}_{args.task_name}_seed_{args.seed}') or args.resplit:
        
        # build a folder
        if not os.path.exists(f'{args.result_path}/{args.data_name}_{args.task_name}_seed_{args.seed}'):
            os.makedirs(f'{args.result_path}/{args.data_name}_{args.task_name}_seed_{args.seed}')

        # split dataset
        if args.data_name == 'mp' and args.task_name == 'formation_energy':
            # train 60000, valid 4619, test 4620, total 69239 in the paper
            train_valid_names, test_names = train_test_split(names, test_size=4620, random_state=args.seed)
            with open(f'{args.result_path}/{args.data_name}_{args.task_name}_seed_{args.seed}/test.txt', 'w') as fw:
                for name in test_names:
                    fw.write(name + '\n')
            for idx in range(args.total_folds):
                train_names, valid_names = train_test_split(train_valid_names, test_size=4619, random_state=args.seed + idx)
                with open(f'{args.result_path}/{args.data_name}_{args.task_name}_seed_{args.seed}/train_fold_{idx + 1}.txt', 'w') as fw:
                    for name in train_names:
                        fw.write(name + '\n')
                with open(f'{args.result_path}/{args.data_name}_{args.task_name}_seed_{args.seed}/valid_fold_{idx + 1}.txt', 'w') as fw:
                    for name in valid_names:
                        fw.write(name + '\n')
        else:
            # 9-fold cv in train and fold, 9 for keeping the same number of valid and test
            train_valid_names, test_names = train_test_split(names, test_size=0.1, random_state=args.seed)
            with open(f'{args.result_path}/{args.data_name}_{args.task_name}_seed_{args.seed}/test.txt', 'w') as fw:
                for name in test_names:
                    fw.write(name + '\n')
            kf = KFold(n_splits=args.total_folds, shuffle=True, random_state=args.seed)
            for idx, (train_index, valid_index) in enumerate(kf.split(train_valid_names)):
                train_names, valid_names = np.array(train_valid_names)[train_index].tolist(), np.array(train_valid_names)[valid_index].tolist()
                with open(f'{args.result_path}/{args.data_name}_{args.task_name}_seed_{args.seed}/train_fold_{idx + 1}.txt', 'w') as fw:
                    for name in train_names:
                        fw.write(name + '\n')
                with open(f'{args.result_path}/{args.data_name}_{args.task_name}_seed_{args.seed}/valid_fold_{idx + 1}.txt', 'w') as fw:
                    for name in valid_names:
                        fw.write(name + '\n')       

    # load names
    assert args.run_fold > 0
    with open(f'{args.result_path}/{args.data_name}_{args.task_name}_seed_{args.seed}/train_fold_{args.run_fold}.txt', 'r') as f:
        train_names = [line.strip() for line in f.readlines()]
    with open(f'{args.result_path}/{args.data_name}_{args.task_name}_seed_{args.seed}/valid_fold_{args.run_fold}.txt', 'r') as f:
        valid_names = [line.strip() for line in f.readlines()]
    with open(f'{args.result_path}/{args.data_name}_{args.task_name}_seed_{args.seed}/test.txt', 'r') as f:
        test_names = [line.strip() for line in f.readlines()]
    
    # task setting
    if args.data_name == 'mp':
        task_map = {'band_gap': 0, 'formation_energy': 1}
        args.task_type, args.task_loss, args.task_metric, args.task_index = 'regression', 'mae', 'mae', task_map[args.task_name]
    elif args.data_name == 'matgen':
        task_map = {'band_gap':0, 'total_energy':1, 'per_atom_energy':2, 'formation_energy': 3, 'efermi': 4, 'magnetization': 5}
        args.task_type, args.task_loss, args.task_metric, args.task_index = 'regression', 'mae', 'mae', task_map[args.task_name]
    elif args.data_name.startswith('demo'):
        args.task_type, args.task_loss, args.task_metric, args.task_index = 'regression', 'mae', 'mae', 0
    else:
        print("Not supported task setting!")
        assert False
    
    # normalize label with the shape of (1, 1)
    if args.task_type == 'regression':
        train_labels = [[labels[name][args.task_index]] for name in train_names]
        label_mean = torch.from_numpy(np.nanmean(train_labels, axis=0, keepdims=True)).float().to(device)
        label_std = torch.from_numpy(np.nanstd(train_labels, axis=0, keepdims=True)).float().to(device)
    else:
        label_mean = torch.from_numpy(np.array([[0]])).long().to(device)
        label_std = torch.from_numpy(np.array([[1]])).long().to(device)
    
    # construct data loader
    train_loader, (args.node_features_dim, args.edge_features_dim) = get_loader(train_names, crystals, graphs, labels, args.task_index,
                                                                                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader, (_, _) = get_loader(valid_names, crystals, graphs, labels, args.task_index,
                                      batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader, (_, _) = get_loader(test_names, crystals, graphs, labels, args.task_index,
                                     batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f'train size: {len(train_names):,} | valid size: {len(valid_names):,} | test size: {len(test_names):,}')
    
    ######################################## model setting area ########################################
    
    # model
    model = CrystalNet(node_features=args.node_features_dim, edge_features=args.edge_features_dim, hidden_features=args.hidden_features_dim, 
                       output_features=1, num_step_message_passing=args.num_step_message_passing).to(device)
    
    # optimizer, scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = NoamLR(optimizer=optimizer, warmup_epochs=[5], total_epochs=[args.max_epochs],steps_per_epoch=len(train_loader),
                       init_lr=[args.learning_rate], max_lr=[args.learning_rate * 5], final_lr=[args.learning_rate])

    # initital weight and print model summary
    initialize_weights(model)
    print(model)
    
    ######################################## training area ########################################
    
    with open(f'{args.result_path}/{args.data_name}_{args.task_name}_seed_{args.seed}/{model.__class__.__name__}_fold_{args.run_fold}.txt', 'w') as f:
        best_epoch = -1
        best_metric = float('inf') if args.task_type == 'regression' else float('-inf')
        best_results = None
        
        for epoch in range(1, args.max_epochs + 1):
            
            # train stage
            model.train()
            train_results = loop(data_loader=train_loader, model=model, optimizer=optimizer, scheduler=scheduler,
                                 loss_name=args.task_loss, metric_name=args.task_metric, data_mean=label_mean, data_std=label_std, device=device)
            
            # valid stage
            model.eval()
            with torch.no_grad():
                valid_results = loop(data_loader=valid_loader, model=model, optimizer=None, scheduler=None,
                                     loss_name=args.task_loss, metric_name=args.task_metric, data_mean=label_mean, data_std=label_std, device=device)
            
            # store model
            if args.task_type == 'regression':
                if valid_results[f'{args.task_metric}'] < best_metric:
                    best_epoch, best_metric, best_results = epoch, valid_results[f'{args.task_metric}'], valid_results
                    torch.save(model.state_dict(), f'{args.result_path}/{args.data_name}_{args.task_name}_seed_{args.seed}/{model.__class__.__name__}_fold_{args.run_fold}.ckpt')
            else:
                if valid_results[f'{args.task_metric}'] > best_metric:
                    best_epoch, best_metric, best_results = epoch, valid_results[f'{args.task_metric}'], valid_results
                    torch.save(model.state_dict(), f'{args.result_path}/{args.data_name}_{args.task_name}_seed_{args.seed}/{model.__class__.__name__}_fold_{args.run_fold}.ckpt')

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
        model = CrystalNet(node_features=args.node_features_dim, edge_features=args.edge_features_dim, hidden_features=args.hidden_features_dim, 
                           output_features=1, num_step_message_passing=args.num_step_message_passing).to(device)
        model.eval()
        with torch.no_grad():
            model.load_state_dict(torch.load(f'{args.result_path}/{args.data_name}_{args.task_name}_seed_{args.seed}/{model.__class__.__name__}_fold_{args.run_fold}.ckpt', map_location=device))
            test_results = loop(data_loader=test_loader, model=model, optimizer=None, scheduler=None,
                                loss_name=args.task_loss, metric_name=args.task_metric, data_mean=label_mean, data_std=label_std, device=device)

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
