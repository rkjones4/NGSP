import ast
import numpy as np
import os
import sys
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from grammar import Grammar
import json
import torch
import utils

DATA_DIR = "TODO_PATH_TO_PARTNET"

def get_area(_v, _f):
    vs = torch.tensor(_v).float().unsqueeze(0)
    faces = torch.tensor(_f).long()
    face_normals = torch.cross(
        vs[:, faces[:, 1], :] - vs[:, faces[:, 0], :],
        vs[:, faces[:, 2], :] - vs[:, faces[:, 1], :],
        dim=2,
    )
    face_areas = torch.norm(face_normals, dim=2) + 1e-8
    
    return face_areas.sum().item()

def get_shape_area(ind):
        
    parts = os.listdir(f'{DATA_DIR}/{ind}/objs')

    parts = [p for p in parts if '.obj' in p]
    
    areas = []
    for p in parts:
        v, f = utils.loadObj(f'{DATA_DIR}/{ind}/objs/{p}')
        areas.append(get_area(v, f))
        
    areas = np.array(areas)
    areas /= areas.sum()

    return areas
    
def getInds():    
    inds = os.listdir(DATA_DIR)
    inds.sort()
    return inds
                      
def main(out_dir):
    inds = getInds()
    os.system(f'mkdir {out_dir}')
    for ind in tqdm(inds):
        area = get_shape_area(ind)
        np.save(f'{out_dir}/area_{ind}.npy', area)

        
if __name__ == '__main__':
    with torch.no_grad():
        main(sys.argv[1])

