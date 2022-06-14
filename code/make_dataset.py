import ast
import numpy as np
import os
import sys
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from grammar import Grammar
import json

DATA_DIR = "TODO_PATH_TO_PARTNET"

# DEBUGGING STUFF
VERBOSE = True
MAX_SHAPES = None

def getInds():    
    inds = os.listdir(DATA_DIR)
    inds.sort()
    return inds
                      
# finds info for all parts with part_name and with parent with parent name if given
# returns obj-names associated with part and all ori_ids
def newParseParts(folder, json, grammar, pre=''):    
    name = pre + json['name']

    m2p = {}

    assert name in grammar.fl2i, f'name {name} not in grammar'
    for m in json['objs']:
        m2p[m] = name
        
    if 'children' in json:
        for c in json['children']:            
            cm2p = newParseParts(folder, c, grammar,name + '/')
            m2p.update(cm2p)
                
    return m2p

        
def parseData(folder, grammar):
    with open(DATA_DIR+folder+"/result_after_merging.json") as f:
        json = ast.literal_eval(f.readline())[0]

    if json['name'] != grammar.start_symbol:
        assert False, 'wrong-cat'
            
    m2p = newParseParts(folder, json, grammar)

    seen_objs = set(m2p.keys())
    all_objs = set([i.split('.')[0] for i in os.listdir(f'{DATA_DIR}{folder}/objs')])
        
    if seen_objs != all_objs:
        assert False, 'missing-obj'

    # Tuples of (label, mesh)
    parts = []
    
    for m, p in m2p.items():                        
        parts.append((grammar.fl2i[p], m))

    return parts

def writeData(data, out_folder, shape):
    os.system(f'mkdir {out_folder}/{shape} > /dev/null 2>&1')
    j = {}
    j['parts'] = [f'{DATA_DIR}/{shape}/objs/{m}.obj' for _,m in data]
    j['labels'] = [l for l,_ in data]
    
    json.dump(j, open(f'{out_folder}/{shape}/data.json', 'w'))

    
def format_data(out_folder, category):
    os.system(f'mkdir {out_folder} > /dev/null 2>&1')
    
    grammar = Grammar(category)    
    print(f"Using Grammar for {category}")    
            
    all_shapes = getInds()        
    
    misses = 0
    count = 1e-8

    seen = os.listdir(out_folder)
    
    for shape in tqdm(all_shapes):

        if MAX_SHAPES is not None and count > MAX_SHAPES:
            break
                
        try:
            data = parseData(shape, grammar)            
            count += 1
            
        except Exception as e:
            if 'wrong-cat' in e.args[0]:
                continue
            
            misses += 1
            count += 1
            
            if 'missing-obj' in e.args[0]:
                if VERBOSE:
                    print(f'failed {shape} with missing a part obj in terminal set')
                continue
            elif 'not in grammar' in e.args[0]:
                if VERBOSE:
                    print(f'failed {shape} with bad label {e.args[0]}')
                continue                    
            else:
                raise e

        writeData(data, out_folder, shape)
        

                
    print(f"Misses: {misses} ({(misses * 1.) / count})")
                
    
if __name__ == '__main__':
    out_dir = sys.argv[1]
    cat = sys.argv[2]

    format_data(out_dir, cat)

