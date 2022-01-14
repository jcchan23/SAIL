#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run.py
@Time    :   2022/01/14 08:45:52
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

blosum_dict = dict()
with open(f'blosum62.txt', 'r') as f:
    lines = f.readlines()[7:]
    for i in range(20):
        line = lines[i].strip().split()
        blosum_dict[line[0]] = [int(num) for num in line[1:21]]
        
sequence = "MQEIYRFIDDAIEADRQRYTDIADQIWDHPETRFEEFWSAEHLASAFIA"
feature = [blosum_dict[amino] for amino in sequence]
print(feature)
