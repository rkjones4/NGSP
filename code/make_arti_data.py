import utils
import os
from grammar import Grammar
import data_utils
import numpy as np
from random import choices, sample
from tqdm import tqdm

import random
import torch

MAX_ITERS = 10000

def make_arti_props(ind, labels, grammar, args):
    os.system(f'mkdir {args.search_data_path}/{ind} > /dev/null 2>&1')
    probs = []
    dists = grammar.label_dists
    probs = (dists.max().item() + 1) - dists
    np.fill_diagonal(probs, 0)
    
    probs = probs / dists.max().item()
    probs /= probs.sum(axis=1).reshape(-1, 1)

    seen = set()

    diffs = []
    props = []

    for diff, num in [
            (0, 1),
            (1, 100),
            (2, 200),
            (3, 300),
            (4, 400),
            (5, 500),
            (int(len(labels) * .1), 500),
            (int(len(labels) * .2), 1000),
            (int(len(labels) * .3), 1500),
            (int(len(labels) * .4), 2000),
            (int(len(labels) * .5), 2500),
            (int(len(labels)), 999),
    ]:
        c = 0
        t = 0
        while c < num:            
            if t > MAX_ITERS:
                break
            
            prop = []
            inds = set(sample(list(range(len(labels))), min(diff, len(labels))))
            for i, l in enumerate(labels):
                if i in inds:
                    prop.append(choices(list(range(probs.shape[0])), probs[l])[0])
                else:
                    prop.append(l)

            sig = tuple(prop)
            t += 1
            if sig not in seen:
                c += 1
                props.append(prop)
                diffs.append(diff)
                seen.add(sig)

    props = np.array(props).astype('int16')
    
    np.savez(
        f'{args.search_data_path}/{ind}/prop_samples.npz',
        prop_samples = props,
    )
    
def main(args):
    grammar = Grammar(args.category)

    random.seed(args.rd_seed)
    np.random.seed(args.rd_seed)
    torch.manual_seed(args.rd_seed)
    
    
    train_data = data_utils.load_dataset(
        f'{args.data_path}/{args.category}',
        f'{args.category}/split.json',
        f'{args.train_name}_{args.train_size}'
    )

    val_data = data_utils.load_dataset(
        f'{args.data_path}/{args.category}',
        f'{args.category}/split.json',
        f'val_{args.eval_size}',
    )

    test_data = data_utils.load_dataset(
        f'{args.data_path}/{args.category}',
        f'{args.category}/split.json',
        f'test_{args.eval_size}',
    )

    for data in [train_data, val_data, test_data]:
        for ind, labels in tqdm(list(zip(data['inds'], data['labels']))):
            make_arti_props(ind, grammar.level_map[labels], grammar, args)
            



if __name__ == '__main__':
    arg_list = utils.SEARCH_DEF_ARGS
    args = utils.getArgs(arg_list)
    assert args.search_data_path is not None
    os.system(f'mkdir {args.search_data_path} > /dev/null 2>&1')
    
    for cat in ['chair','table','lamp','vase','storagefurniture','knife']:
        print(cat)
        args.category = cat
        main(args)
