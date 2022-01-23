#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run.py
@Time    :   2022/01/14 08:45:08
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

aaphy7_dict = dict()
with open(f'./Features/aaphy7/aaphy7.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        aaphy7_dict[line[0]] = [float(num) for num in line[1:]]

sequence = "MQEIYRFIDDAIEADRQRYTDIADQIWDHPETRFEEFWSAEHLASAFIA"
feature = [aaphy7_dict[amino] for amino in sequence]
print(feature)