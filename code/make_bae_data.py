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

import bae_net.gen_data as gd

def combMesh(meshes):
    verts = []
    faces = []
        
    offset = 0
        
    for si, (_v,_f) in enumerate(meshes):
        v = torch.from_numpy(_v)
        f = torch.from_numpy(_f)
        faces.append(f + offset)
        verts.append(v)
        offset += v.shape[0]
            
    verts = torch.cat(verts, dim=0).float().to(device)
    faces = torch.cat(faces, dim=0).long().to(device)
    
    return verts.cpu().numpy(), faces.cpu().numpy()
    


def main(args):
    
    data_inds = []

    path = f'{args.data_path}/{args.category}'

    # Indentation helper 
    if True:

        split_file = f'{args.category}/split.json'
        split_info = json.load(open(f'data_splits/{split_file}'))
    
        train_inds = [int(s) for s in split_info['train'][:args.train_size]]
        val_inds = [int(s) for s in split_info['val'][:args.eval_size]]
        test_inds = [int(s) for s in split_info['test'][:args.eval_size]]

        uinds = []
        
        for fn in os.listdir(path):
            try:
                uinds.append(int(fn))
            except Exception:
                pass


        uinds = set(uinds)
        uinds = uinds - set(train_inds)
        uinds = uinds - set(val_inds)
        uinds = uinds - set(test_inds)

        uinds = list(uinds)
        uinds.sort()

        uinds = uinds[:args.unsup_size]

        data_inds = data_inds + train_inds + val_inds + test_inds + uinds

    data_inds = set(data_inds)

    print(f'Number of inds : {len(data_inds)}')
    
    for ind in tqdm(data_inds):
        d = data_utils.load_data(path, ind)
        meshes = d['meshes']
        verts, faces  = combMesh(
            meshes
        )
        _ = gd.load_baenet_data(ind, verts, faces)
            
if __name__ == '__main__':
    arg_list = [
        ('-us', '--unsup_size', None,  int),
    ]
    
    args = utils.getArgs(arg_list)
    main(args)
    
