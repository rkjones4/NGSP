import ast
import numpy as np
import os
import sys
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from grammar import Grammar
import utils, torch
from random import shuffle
import json
from make_dataset import DATA_DIR

PARSED_DIR = '../data'
OUT_DIR = 'data_splits'

DO_SMART_SPLIT = True

# Each segment must be bigger than this, or we reject the shape
PERC_CUTOFF = 0.001

VERBOSE = True
CHECK_AREA = True

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

def is_valid(in_dir, ind, grammar):
    folder = f'{in_dir}/{ind}'
    
    j = json.load(open(f'{folder}/data.json'))
    seg_labels = j['labels']
    level_labels = grammar.level_map[seg_labels]
    
    if (level_labels < 0).any():        
        return False
            
    return True    

def get_possible_inds(in_dir):
    pos_inds = []
    print("Filtering shape on Area")
    for ind in tqdm(list(os.listdir(in_dir))):
        if '.json' in ind:
            continue

        if CHECK_AREA:
            areas = np.load(f'{PARSED_DIR}/area/area_{ind}.npy')
            if areas.min() < PERC_CUTOFF:
                continue

        pos_inds.append(ind)

    return pos_inds
        
def get_valid_inds(in_dir, pos_inds, grammar):

    valid_inds = []

    seen_shape_ids = set()
    
    for ind in tqdm(pos_inds):
            
        if not is_valid(in_dir, ind, grammar):
            continue
        
        shape_id = json.load(open(f'{DATA_DIR}/{ind}/meta.json'))['model_id']

        if shape_id in seen_shape_ids:
            continue

        seen_shape_ids.add(shape_id)
        valid_inds.append(ind)
        
    return valid_inds

def getAssocTerms(folder, grammar):
    seg_labels = json.load(open(f'{folder}/data.json'))['labels']
    seg_labels = grammar.level_map[seg_labels]
    return set(np.unique(seg_labels).tolist())
    
def getIndToTermMap(in_dir, grammar, inds):
    m = {}
    for ind in inds:
        m[ind] = getAssocTerms(f'{in_dir}/{ind}', grammar)
        
    return m

def smart_split(in_dir, pos_inds, grammar):
    inds = get_valid_inds(in_dir, pos_inds, grammar)
    i2term = getIndToTermMap(in_dir, grammar, inds)

    test_inds = []
    val_inds = []
    train_inds = []

    if len(inds) < 150:
        print(f"Too few inds {len(inds)}")
        return None

    elif len(inds) < 500:
        EVAL_SIZE = 50
        TRAIN_SIZE = len(inds) - 100

    else:
        EVAL_SIZE = min(int(0.1 * len(inds)), 400)
        TRAIN_SIZE = min(len(inds) - (2 * EVAL_SIZE), 2000)
        
    test_counts = np.zeros(len(grammar.terminals)).astype('long')
    val_counts = np.zeros(len(grammar.terminals)).astype('long')
    train_counts = np.zeros(len(grammar.terminals)).astype('long')
    
    while len(test_inds) < EVAL_SIZE or \
          len(val_inds) < EVAL_SIZE or \
          len(train_inds) < TRAIN_SIZE:

        test_min = test_counts.min()
        val_min = val_counts.min()
        train_min = train_counts.min()
        
        if len(test_inds) >= EVAL_SIZE:
            test_min = 1e8

        if len(val_inds) >= EVAL_SIZE:
            val_min = 1e8

        if len(train_inds) >= TRAIN_SIZE:
            train_min = 1e8

        all_min = min([test_min, val_min, train_min])

        if test_min == all_min:
            cur_counts = test_counts
            cur_inds = test_inds

        elif val_min == all_min:
            cur_counts = val_counts
            cur_inds = val_inds

        elif train_min == all_min:
            cur_counts = train_counts
            cur_inds = train_inds

        goal_terms = set((cur_counts == all_min).nonzero()[0].tolist())

        best_ind = None
        best_inter = 0

        for ind, terms in i2term.items():
            inter = len(goal_terms.intersection(terms))
            if inter == len(goal_terms):
                best_ind = ind
                break

            if inter > best_inter:
                best_ind = ind
                best_inter = inter

        if best_ind is None:
            print(
                f"Ran out of {goal_terms} :"
                f" {[grammar.i2l[g] for g in goal_terms]}"
            )
            for g in goal_terms:
                train_counts[g] += TRAIN_SIZE
                val_counts[g] += TRAIN_SIZE
                test_counts[g] += TRAIN_SIZE

            continue
            
                
        cur_terms = i2term.pop(best_ind)                
        cur_inds.append(best_ind)

        for c in cur_terms:
            cur_counts[c] += 1
    
    split_map = {
        'test': test_inds,
        'val': val_inds,
        'train': train_inds
    }
        
    return split_map
        

def split_data(in_dir, cat, name):

    pos_inds = get_possible_inds(in_dir)

    # Indent helper
    if True:
        grammar = Grammar(cat)
                
        split_map = smart_split(in_dir, pos_inds, grammar)
        if split_map is None:
            continue
        
        print(f"Train/Val/Test : ({len(split_map['train'])}/"
              f"{len(split_map['val'])}/{len(split_map['test'])})")

        os.system(f'mkdir {OUT_DIR}/{cat} > /dev/null 2>&1')                
        os.system(f'rm {OUT_DIR}/{cat}/{name}.json > /dev/null 2>&1')
        json.dump(split_map, open(f'{OUT_DIR}/{cat}/{name}.json', 'w')) 
        
if __name__ == '__main__':
    in_dir = sys.argv[1]
    cat = sys.argv[2]
    name = sys.argv[3]
    with torch.no_grad():
        split_data(in_dir, cat, name)
