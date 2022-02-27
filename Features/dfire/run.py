#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run.py
@Time    :   2022/01/23 20:09:20
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

import os
from tqdm import tqdm

for name in tqdm(os.listdir(f'./input')):
    if name.endswith('.pdb'):
        name = name.split('.')[0]
        # theta
        os.system(f'./dfire_rotamer ./input/{name}.pdb > ./output/{name}.dfire')
