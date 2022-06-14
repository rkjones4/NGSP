from lik_mods.sem_label_lik import SemLik
from lik_mods.reg_group_lik import RegLik

import utils
import os
from grammar import Grammar
import data_utils
import numpy as np
from random import choices, sample
from tqdm import tqdm
import eval_utils
import torch
from utils import device

def sampleShape(meshes, seg_labels):
    if True:
        verts = []
        faces = []
        segments = []
        labels = []
        
        offset = 0
        
        for si, ((_v,_f), sl) in enumerate(zip(meshes, seg_labels)):
            v = torch.from_numpy(_v)
            f = torch.from_numpy(_f)
            faces.append(f + offset)
            verts.append(v)
            segments += [si for _ in range(f.shape[0])]
            labels += [sl for _ in range(f.shape[0])]
            offset += v.shape[0]
            
        verts = torch.cat(verts, dim=0).float().to(device)
        faces = torch.cat(faces, dim=0).long().to(device)
        segments = torch.tensor(segments).long()
        labels = torch.tensor(labels).long()
        
        samps, face_inds, normals = utils.sample_surface(
            faces, verts.unsqueeze(0), utils.NUM_PROP_INPUT_PTS
        )
            
        samps = samps.cpu()
        face_inds.cpu()
        
        samp_segments = segments[face_inds]    
        samp_labels = labels[face_inds]
        
        return samps, samp_segments, samp_labels

def calc_seg_miou(A_seg_preds, data, grammar):
    A_labels = []
    A_preds = []
    
    for meshes, labels, seg_preds in \
        zip(data['meshes'], data['labels'], A_seg_preds):
        seg_labels = grammar.level_map[labels]

        _, samp_segments, samp_labels = sampleShape(
            meshes,
            seg_labels
        )

        samp_preds = torch.zeros(samp_labels.shape).long() - 1
        for i, p in enumerate(seg_preds):
            inds = (samp_segments == i).nonzero().flatten()
            samp_preds[inds] = p.item()

        assert (samp_preds >= 0).all(), 'some -1 label left'

        A_labels.append(samp_labels.cpu())
        A_preds.append(samp_preds.cpu())

    iou = eval_utils.calc_mIoU(A_labels, A_preds, grammar)
    return iou
        
def calc_metrics(lik_preds, prop_preds, diffs, test_data, grammar, args, lik_name):
    res = {
        'first_gt': [],        
        'first_diff': [],
        'pos_gt': []
    }
    seg_preds = []
    
    utils.log_print(f"\nLik Res {lik_name} : ", args, 'res')    
    
    torch.save(lik_preds, f'{args.outpath}/{args.exp_name}/{lik_name}.pt')
    
    for lp, pp, df in zip(lik_preds, prop_preds, diffs):

        bind = lp.argmax().item()

        if bind == 0:
            res['first_gt'].append(1.)
        else:
            res['first_gt'].append(0.)

        res['first_diff'].append(float(df[bind]))

        res['pos_gt'].append((lp.shape[0]-1) - float(lp.argsort().argsort()[0]))
        
        seg_preds.append(pp[bind])

    res = {k:round(torch.tensor(v).mean().item(),3) for k,v in res.items()}
        
    res['first_miou'] = calc_seg_miou(seg_preds, test_data, grammar)

    for k,v in res.items():
        utils.log_print(f'  {k} : {v}', args, 'res')
    

def eval_lik(args):

    grammar = Grammar(args.category)
    args.outpath = 'lik_mods/model_output'
    args.lik_model_outpath = args.outpath
    
    if args.lik_mode == 'sem':
        lik = SemLik(args, grammar)
    elif args.lik_mode == 'reg':
        lik = RegLik(args, grammar)
    else:
        assert False    
            
    lik.train()

    
if __name__ == '__main__':
    arg_list = [
        ('-lm', '--lik_mode', None, str),
        ('-ebs', '--eval_batch_size', 100, int),
        ('-mns', '--max_num_samps', 10000, int),
        ('-len', '--load_exp_name', None, str),
        ('-esdp', '--eval_search_data_path', None, str)
    ] + utils.SEARCH_DEF_ARGS
    
    args = utils.getArgs(arg_list)
    assert args.search_data_path is not None
    assert args.lik_mode is not None

    if args.load_exp_name is None:
        args.load_exp_name = args.exp_name
    else:
        assert args.do_train == 'n'

    if args.eval_search_data_path is None:
        args.eval_search_data_path = args.search_data_path
    
    eval_lik(args)
