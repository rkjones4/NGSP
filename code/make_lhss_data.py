from grammar import Grammar
import sys, os, torch
import numpy as np
import argparse
import data_utils
import utils
from torch.utils.data import DataLoader
import train_utils, eval_utils
from utils import device
import ast
from copy import deepcopy, copy
import random
from tqdm import tqdm
import pickle
import json
import time

import lhss.feat_code.calc_feat as cf

def main(args):
    
    data_inds = []

    path = f'{args.data_path}/{args.category}'
    
    split_file = f'{args.category}/split.json'
    split_info = json.load(open(f'data_splits/{split_file}'))    
    train_inds = [int(s) for s in split_info[args.set_name][:args.eval_size]]        
    data_inds = train_inds

    data_inds = set(data_inds)

    print(f'Number of inds : {len(data_inds)}')

    T = .01
    c = .01
    
    for count,ind in enumerate(data_inds):
        t = time.time()
        print(f'Prog ({count}/{len(data_inds)}) | Avg Time {T/c}')
        d = data_utils.load_data(path, ind)
        meshes = d['meshes']        
        _ = cf.load_lhss_data(meshes, ind)
        T += time.time() -t
        c += 1.
            
if __name__ == '__main__':
    arg_list = [
        ('-set', '--set_name', None, str),
    ]
    
    args = utils.getArgs(arg_list)
    assert args.set_name in ['train', 'val', 'test']
    with torch.no_grad():
        main(args)
    
