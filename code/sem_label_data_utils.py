import numpy as np
import torch
import os
import utils
from tqdm import tqdm
from utils import device
import json
from random import sample, randint
from copy import deepcopy

                        
def check_valid_neg(
    area_data,
    pos_sig,
    neg_sig,
    args,
    name
):    
    
    area_sim = calc_area_sim(area_data, pos_sig, neg_sig)

    if area_sim > args.max_neg_area_sim:
        return False
    else:
        return True

    assert False, 'surprisingly, something went wrong ;('
        

    
def calc_area_sim(area_data, pos_sig, neg_sig):
    match_area = 1e-8
    pos_area = 1e-8
    neg_area = 1e-8

    info = {}

    for sig in pos_sig:
        if isinstance(sig, tuple):
            seg = sig[0]
            key = sig[1]                        
        else:
            seg = sig
            key = True
            
        info[seg] = key
        pos_area += area_data[seg]

    for sig in neg_sig:
        if isinstance(sig, tuple):
            seg = sig[0]
            key = sig[1]
        else:
            seg = sig
            key = True

        neg_area += area_data[seg]
        
        if seg in info and info[seg] == key:
            match_area += area_data[seg]

    ref_area = max(pos_area, neg_area)

    return match_area / ref_area
        
def getSig(parts, labels, name):
    if 'lay' in name:
        z = list(zip(parts, labels))
    elif 'geom' in name:
        z = deepcopy(parts)            
    z.sort()
    return tuple(z)


def parseDataWithNode(search_data, node, grammar, args, name):
    data = {
        'pos_inds': [],
        'pos_in_parts': [],
        'pos_in_cls': [],        

        'prop_inds': [],
        'prop_in_parts': [],
        'prop_in_cls': [],

        'areas': [],

        'hn_inds': [],
        'hn_prop_inds': [],
        'hn_prop_in_parts': [],
        'hn_prop_in_cls': [],
        
    }
    
    n2cl = grammar.n2cl[node]
    
    labels = search_data['labels']
    l = list(zip(search_data['inds'], search_data['labels'], search_data['areas']))
    
    for i, (ind, labels, area_data) in tqdm(enumerate(l),total=len(l)):
        
        cls = n2cl[labels]

        in_parts = (cls >= 0).nonzero()[0].reshape(-1)

        HARD_NEG = False
        
        if in_parts.shape[0] > 0:
            data['pos_inds'].append(i)
            data['pos_in_parts'].append(in_parts)
            data['pos_in_cls'].append(cls[in_parts])
            data['areas'].append(area_data)
        else:
            HARD_NEG = True
            data['hn_inds'].append(i)            

        pos_sig = getSig(in_parts.tolist(), cls[in_parts].tolist(), name)

        seen = set([pos_sig])
            
        search_data = np.load(
            f'{args.search_data_path}/{ind}/prop_samples.npz'
        )
        prop_samples = search_data['prop_samples']
        
        for ps in prop_samples[:args.max_prop_negs]:

            fps = np.array([grammar.fl2i[grammar.i2l[l]] for l in ps])

            ps_cls = n2cl[fps]

            ps_in_parts = (ps_cls >= 0).nonzero()[0].reshape(-1)
            ps_in_cls = ps_cls[ps_in_parts]

            if ps_in_parts.shape[0] == 0:
                continue

            neg_sig = getSig(ps_in_parts.tolist(), ps_in_cls.tolist(), name)
                                   
            if neg_sig not in seen:

                seen.add(neg_sig)
                
                valid_neg = check_valid_neg(
                    area_data,
                    pos_sig,
                    neg_sig,
                    args,
                    name
                )

                if HARD_NEG:
                    assert valid_neg, 'unexpected'
                
                if not valid_neg:
                    continue

                if not HARD_NEG:
                    data['prop_inds'].append(i)
                    data['prop_in_parts'].append(ps_in_parts)
                    data['prop_in_cls'].append(ps_in_cls)
                else:
                    data['hn_prop_inds'].append(i)
                    data['hn_prop_in_parts'].append(ps_in_parts)
                    data['hn_prop_in_cls'].append(ps_in_cls)
    
    return data

