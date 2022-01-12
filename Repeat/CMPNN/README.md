# CMPNN-dgl

Reimplementation of the CMPNN model by using dgl.

The original implementation of CMPNN could be referred at [CMPNN](https://github.com/SY575/CMPNN).

The IJCAI 2020 paper could be referred at [Communicative Representation Learning on Attributed Molecular Graphs](https://www.ijcai.org/Proceedings/2020/0392.pdf).

## Dependencies
+ cuda == 10.2
+ cudnn == 7.6.5
+ dgl-cu10.2 == 0.7.2
+ numpy == 1.19.1
+ pandas == 1.1.0
+ python == 3.7.7
+ RDKit == 2020.03.4
+ scikit-learn == 0.23.2
+ torch == 1.8.1
+ tqdm == 4.48.2

## Overview

We report three different dataset split types and their corresponding results.

### 1. Scaffold Split

This result could be refered from [GROVER](https://arxiv.org/pdf/2007.02835.pdf), [MPG](https://arxiv.org/pdf/2012.11175.pdf),  [CoMPT](https://www.ijcai.org/proceedings/2021/0309.pdf) and [KCL](https://arxiv.org/pdf/2112.00544.pdf). Thanks a lot for their working! 

| Dataset | BBBP | Tox21 | Toxcast | Sider | ClinTox | ESOL | FreeSolv | Lipophilicity |
|:---:    |:---: |:---:  |:---:    |:---:  |:---:    |:---: |:---:     |:---:          |
|Molecules|2039  |7831   |8575     |1427   |1478     |1128  |642       |4200           |
|Task     |1 GC  |12 GC  |617 GC   |27 GC  |2 GC     |1 GR  |1 GR      |1 GR           |
|Metrics  |AUROC |AUROC  |AUROC    |AUROC  |AUROC    |RMSE  |RMSE      |RMSE           |
|GCN      |0.877±0.036|0.772±0.041|0.650±0.025|0.593±0.034|0.845±0.051|1.068±0.050|2.900±0.135|0.712±0.049|
|MPNN     |0.913±0.041|0.808±0.024|0.691±0.013|0.595±0.030|0.879±0.054|1.167±0.430|2.185±0.952|0.672±0.051|
|DMPNN    |0.919±0.030|0.826±0.023|0.718±0.011|0.632±0.023|0.897±0.040|0.980±0.258|2.177±0.914|0.653±0.046|
|CMPNN    |0.927±0.017|0.806±0.016|0.738±0.010|0.616±0.003|0.902±0.008|0.798±0.112|2.007±0.442|0.614±0.029|
|CMPNN-dgl|0.955±0.011|0.796±0.007|0.715±0.013|0.624±0.025|0.873±0.044|0.774±0.096|1.780±0.656|0.607±0.038|

### 2. Random Split
This result could be refered from [CMPNN](https://github.com/SY575/CMPNN).
| Dataset | BBBP | Tox21 | Toxcast | Sider | ClinTox | ESOL | FreeSolv | Lipophilicity |
|:---:    |:---: |:---:  |:---:    |:---:  |:---:    |:---: |:---:     |:---:          |
|Molecules|2039  |7831   |8575     |1427   |1478     |1128  |642       |4200           |
|Task     |1 GC  |12 GC  |617 GC   |27 GC  |2 GC     |1 GR  |1 GR      |1 GR           |
|Metrics  |AUROC |AUROC  |AUROC    |AUROC  |AUROC    |RMSE  |RMSE      |RMSE           |
|GCN      |0.690      |0.829      |-          |0.638      |0.807      |0.970      |1.400      |-          |
|MPNN     |0.910±0.032|0.844±0.014|-          |0.641±0.014|0.881±0.037|0.702±0.042|1.242±0.249|-          |
|DMPNN    |0.917±0.037|0.854±0.012|-          |0.658±0.020|0.897±0.042|0.587±0.060|1.009±0.207|-          |
|CMPNN    |0.963±0.003|0.856±0.007|-          |0.666±0.007|0.933±0.012|0.547±0.011|0.819±0.147|-          |
|CMPNN-dgl|0.961±0.012|0.854±0.009|0.755±0.013|0.650±0.021|0.916±0.040|0.601±0.088|0.967±0.122|0.580±0.050|

### 3. 5-fold CV Split
This result could be refered from [MolRep](https://github.com/biomed-AI/MolRep).
| Dataset | BBBP | Tox21 | Toxcast | Sider | ClinTox | ESOL | FreeSolv | Lipophilicity |
|:---:    |:---: |:---:  |:---:    |:---:  |:---:    |:---: |:---:     |:---:          |
|Molecules|2039  |7831   |8575     |1427   |1478     |1128  |642       |4200           |
|Task     |1 GC  |12 GC  |617 GC   |27 GC  |2 GC     |1 GR  |1 GR      |1 GR           |
|Metrics  |AUROC |AUROC  |AUROC    |AUROC  |AUROC    |RMSE  |RMSE      |RMSE           |
|GCN      |0.875±0.036|0.818±0.003|-          |0.590±0.000|0.884±0.004|-          |-          |-          |
|MPNN     |0.932±0.031|0.840±0.014|-          |0.631±0.012|0.841±0.029|-          |-          |-          |
|DMPNN    |0.956±0.007|0.842±0.039|-          |0.638±0.033|0.869±0.005|-          |-          |-          |
|CMPNN    |0.985±0.021|0.859±0.008|-          |0.658±0.002|0.917±0.006|-          |-          |-          |
|CMPNN-dgl|0.969±0.008|0.852±0.008|0.757±0.008|0.649±0.013|0.935±0.038|0.562±0.031|0.945±0.094|0.575±0.025|

*Note*

*(1) GC=Graph Classification, GR=Graph Regression*

*(2) The FreeSolv dataset may need to more works on tuning the hyper-parameters since there are only 642 molecules in this dataset.*

*(3) All split types follow the ratio of 0.8/0.1/0.1 in train/valid/test. The different between random split and 5-fold CV split is validation == test in 3.*

## Running

To reproduce all the results, run firstly:

`python dataset.py`

it will generate a pickle file in the `data/preprocess` with the same dataset name, this pickle file contain 4 objects:

+ `smile_list:` All SMILES strings in the dataset.
+ `mol_dict:` Unique SMILES strings -> RDKit mol object.
+ `graph_dict:` Unique SMILES strings -> dgl graph object.
+ `label_dict:` Unique SMILES strings -> label list.

Then run:

`python run.py --gpu <gpu id> --data_name <dataset> --split_type <split> --run_fold <fold_num>`

+ `<gpu id>` is the gpu id.

+ `<dataset>` is the dataset name, we provide 8 datasets that mentioned in the overview, more datasets and their results will be updated.

+ `<split>` is the split type name, we provide `[scaffold, random, cv]` in the code.

+ `<fold_num>` is the fold number, if you choose cv split, then you must choose fold number from `[1, 2, 3, 4, 5]` since the 5-fold cv.

Others parameters could be refered in the `run.py`.

After running the code, it will create a folder with the format `<args.data_name>_split_<args.split_type>_seed_<args.seed>` in the `./result/` folder.

If choose scaffold split or random split, the folder will contain:
```
├── result
│   ├── bbbp_split_random_seed_2021
│   │   ├── CMPNN_fold_0.ckpt
│   │   ├── CMPNN_fold_0.txt
│   │   ├── test.pickle
│   │   ├── train.pickle
│   │   └── valid.pickle
│   ├── bbbp_split_scaffold_seed_2021
│   │   ├── CMPNN_fold_0.ckpt
│   │   ├── CMPNN_fold_0.txt
│   │   ├── test.pickle
│   │   ├── train.pickle
│   │   └── valid.pickle
```

If choose cv split, the folder will contain:
```
├── result
│   ├── bbbp_split_cv_seed_2021
│   │   ├── CMPNN_fold_1.ckpt
│   │   ├── CMPNN_fold_1.txt
│   │   ├── CMPNN_fold_2.ckpt
│   │   ├── CMPNN_fold_2.txt
│   │   ├── CMPNN_fold_3.ckpt
│   │   ├── CMPNN_fold_3.txt
│   │   ├── CMPNN_fold_4.ckpt
│   │   ├── CMPNN_fold_4.txt
│   │   ├── CMPNN_fold_5.ckpt
│   │   ├── CMPNN_fold_5.txt
│   │   ├── test_fold_1.pickle
│   │   ├── test_fold_2.pickle
│   │   ├── test_fold_3.pickle
│   │   ├── test_fold_4.pickle
│   │   ├── test_fold_5.pickle
│   │   ├── train_fold_1.pickle
│   │   ├── train_fold_2.pickle
│   │   ├── train_fold_3.pickle
│   │   ├── train_fold_4.pickle
│   │   ├── train_fold_5.pickle
│   │   ├── valid_fold_1.pickle
│   │   ├── valid_fold_2.pickle
│   │   ├── valid_fold_3.pickle
│   │   ├── valid_fold_4.pickle
│   │   └── valid_fold_5.pickle
```

## Citation:

Please cite the following paper if you use this code in your work.
```bibtex
@inproceedings{ijcai2020-392,
  title     = {Communicative Representation Learning on Attributed Molecular Graphs},
  author    = {Song, Ying and Zheng, Shuangjia and Niu, Zhangming and Fu, Zhang-hua and Lu, Yutong and Yang, Yuedong},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere}	
  pages     = {2831--2838},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/392},
  url       = {https://doi.org/10.24963/ijcai.2020/392},
}
```
