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
parser.add_argument('--total_folds', type=int, default=5,
                    help='The total folds in cv')
parser.add_argument('--run_fold', type=int, default=0,
                    help='The parallel running fold')
# dataset setting
parser.add_argument('--data_path', type=str, default='./data/preprocess',
                    help='The full path of features of the data.')
parser.add_argument('--data_name', type=str, default='bbbp',
                    help='the dataset name')
parser.add_argument('--split_type', type=str, default='scaffold',
                    help="the dataset split type")
parser.add_argument('--resplit', action='store_true', default=False,
                    help="resplit the dataset with different comments")
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
    
    # load total dataset
    with open(f'{args.data_path}/{args.data_name}.pickle', 'rb') as f:
        # names = list(), others = dict() with key in names
        smiles, mols, graphs, labels = pickle.load(f)
    
    # split dataset(if need!)
    if not os.path.exists(f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}') or args.resplit:
        
        # build a folder
        if not os.path.exists(f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}'):
            os.makedirs(f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}')
        
        # scaffold split
        if args.split_type == 'scaffold':
            train_smiles, valid_smiles, test_smiles = scaffold_split(smiles, frac=[0.8, 0.1, 0.1], balanced=True, include_chirality=False, ramdom_state=args.seed)
        # random split
        elif args.split_type == 'random':
            train_smiles, valid_smiles, test_smiles = random_split(smiles, frac=[0.8, 0.1, 0.1], random_state=args.seed)
        # cv split with valid == test           
        elif args.split_type == 'cv':
            kf = KFold(n_splits=args.total_folds, shuffle=True, random_state=args.seed)
            for idx, (train_index, test_index) in enumerate(kf.split(smiles)):
                train_smiles, test_smiles = np.array(smiles)[train_index].tolist(), np.array(smiles)[test_index].tolist()
                # use pickle instead of txt since there are various smiles that correspond to the same molecule
                with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}/train_fold_{idx + 1}.pickle', 'wb') as fw:
                    pickle.dump(train_smiles, fw)
                with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}/valid_fold_{idx + 1}.pickle', 'wb') as fw:
                    pickle.dump(test_smiles, fw)
                with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}/test_fold_{idx + 1}.pickle', 'wb') as fw:
                    pickle.dump(test_smiles, fw)
        else:
            raise "not supported split type, please refer the split type"

        if args.split_type != 'cv':
            # use pickle instead of txt since there are various smiles that correspond to the same molecule
            with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}/train.pickle', 'wb') as fw:
                pickle.dump(train_smiles, fw)
            with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}/valid.pickle', 'wb') as fw:
                pickle.dump(valid_smiles, fw)
            with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}/test.pickle', 'wb') as fw:
                pickle.dump(test_smiles, fw)
    
    # load pickle
    if args.split_type in ['scaffold', 'random']:
        with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}/train.pickle', 'rb') as f:
            train_smiles = pickle.load(f)
        with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}/valid.pickle', 'rb') as f:
            valid_smiles = pickle.load(f)
        with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}/test.pickle', 'rb') as f:
            test_smiles = pickle.load(f)
    elif args.split_type == 'cv':
        assert args.run_fold > 0
        with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}/train_fold_{args.run_fold}.pickle', 'rb') as f:
            train_smiles = pickle.load(f)
        with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}/valid_fold_{args.run_fold}.pickle', 'rb') as f:
            valid_smiles = pickle.load(f)
        with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}/test_fold_{args.run_fold}.pickle', 'rb') as f:
            test_smiles = pickle.load(f)
    else:
        raise "not supported split type, please refer the split type"
        
    # task setting
    if args.data_name in ['bbbp', 'clintox', 'sider', 'tox21', 'toxcast']:
        args.task_type, args.task_loss, args.task_metric = 'classification','bce', 'auc'
    elif args.data_name in ['esol', 'freesolv', 'lipophilicity']:
        args.task_type, args.task_loss, args.task_metric = 'regression','mse', 'rmse'
    elif args.data_name in ['qm7', 'qm8', 'qm9']:
        args.task_type, args.task_loss, args.task_metric = 'regression','mse', 'mae'
    else:
        raise "Not supported task setting, please refer the correct data name!"
    
    args.task_number = len(labels[train_smiles[0]])
    
    # normalize label with the shape of (1, task_number)
    if args.task_type == 'regression':
        train_labels = [labels[smile] for smile in train_smiles]
        label_mean = torch.from_numpy(np.nanmean(train_labels, axis=0, keepdims=True)).float().to(device)
        label_std = torch.from_numpy(np.nanstd(train_labels, axis=0, keepdims=True)).float().to(device)
    else:
        label_mean = torch.from_numpy(np.array([[0 for _ in range(args.task_number)]])).long().to(device)
        label_std = torch.from_numpy(np.array([[1 for _ in range(args.task_number)]])).long().to(device)
    
    # construct data loader
    train_loader, (args.node_features_dim, args.edge_features_dim) = get_loader(train_smiles, mols, graphs, labels, 
                                                                                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader, (_, _) = get_loader(valid_smiles, mols, graphs, labels, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader, (_, _) = get_loader(test_smiles, mols, graphs, labels, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f'train size: {len(train_smiles):,} | valid size: {len(valid_smiles):,} | test size: {len(test_smiles):,}')
    
    ######################################## model setting area ########################################
    
    # model
    model = CMPNN(node_features=args.node_features_dim, edge_features=args.edge_features_dim, hidden_features=args.hidden_features_dim, 
                  output_features=args.task_number, num_step_message_passing=args.num_step_message_passing).to(device)
    
    # optimizer, scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = NoamLR(optimizer=optimizer, warmup_epochs=[2], total_epochs=[args.max_epochs],steps_per_epoch=len(train_loader),
                       init_lr=[args.learning_rate], max_lr=[args.learning_rate * 10], final_lr=[args.learning_rate])

    # initital weight and print model summary
    initialize_weights(model)
    print(model)
    
    ######################################## training area ########################################
    
    with open(f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}/{model.__class__.__name__}_fold_{args.run_fold}.txt', 'w') as f:
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
                    torch.save(model.state_dict(), f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}/{model.__class__.__name__}_fold_{args.run_fold}.ckpt')
            else:
                if valid_results[f'{args.task_metric}'] > best_metric:
                    best_epoch, best_metric, best_results = epoch, valid_results[f'{args.task_metric}'], valid_results
                    torch.save(model.state_dict(), f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}/{model.__class__.__name__}_fold_{args.run_fold}.ckpt')

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
        model = CMPNN(node_features=args.node_features_dim, edge_features=args.edge_features_dim, hidden_features=args.hidden_features_dim, 
                output_features=args.task_number, num_step_message_passing=args.num_step_message_passing).to(device)
        model.eval()
        with torch.no_grad():
            model.load_state_dict(torch.load(f'{args.result_path}/{args.data_name}_split_{args.split_type}_seed_{args.seed}/{model.__class__.__name__}_fold_{args.run_fold}.ckpt', 
                                             map_location=device))
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
