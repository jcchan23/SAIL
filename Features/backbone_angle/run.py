#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run.py
@Time    :   2022/01/23 20:08:59
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
        os.system(f'./caltheta ./input/{name}.pdb > ./output/theta/{name}.theta')
        # phi psi omega
        os.system(f'./calphipsiomega ./input/{name}.pdb > ./output/ppo/{name}.ppo')
