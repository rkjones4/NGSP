from lik_mods.sem_label_lik import SemLik
from lik_mods.reg_group_lik import RegLik

from models import PointNetPPCls
import train_guide_net as tfs
from math import exp, log
import random
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
import time
        
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
    
    for samp_segments, samp_labels, seg_preds in zip(
        data['samp_segments'], data['samp_labels'], A_seg_preds
    ):
    
        samp_preds = torch.zeros(samp_labels.shape).long() - 1
        for i, p in enumerate(seg_preds):
            inds = (samp_segments == i).nonzero().flatten()
            samp_preds[inds] = p.item()

        assert (samp_preds >= 0).all(), 'some -1 label left'

        A_labels.append(samp_labels.cpu())
        A_preds.append(samp_preds.cpu())

    iou = eval_utils.calc_mIoU(A_labels, A_preds, grammar)
    return iou    
        
def calc_metrics(seg_preds, test_data, grammar, args):
    mIoU = calc_seg_miou(seg_preds, test_data, grammar)
    return mIoU    

def loadPropInfo(prop_net, test_data, args, gramar):

    pn_dists = []
    beams = []
    
    for ind in range(len(test_data['inds'])):
        samps = test_data['samps'][ind]
        segments = test_data['samp_segments'][ind]
        num_segments = test_data['labels'][ind].shape[0]

        inp = {
            'net': prop_net,
            'points': samps[:utils.NUM_PROP_INPUT_PTS].to(device),
            'segments': segments[:utils.NUM_PROP_INPUT_PTS].to(device),
            'num_segments': num_segments,
            'num_samples': args.beam_size,
            'keep_ll': True,
            'batch_size': args.batch_size,
            'mode': 'sample',
            'keep_dist': True,
        }

        samples, seg_dist  = tfs.model_eval(inp)

        rps = torch.stack(
            [s for _,s in samples],
            dim=0
        ).long().cpu()
        
        if rps.shape[0] < args.beam_size:
            rps = torch.cat((
                rps, torch.zeros(args.beam_size-rps.shape[0], rps.shape[1]),
                ), dim = 0
            )

        pn_dists.append(seg_dist.cpu())
        beams.append(rps[:args.beam_size].numpy())
                
    prop_net.to(torch.device('cpu'))
    
    return pn_dists, beams

def loadUniformSamples(test_data, args, grammar):

    beams = []
    num_labels = len(grammar.terminals)
    for ind in range(len(test_data['inds'])):
        num_segments = test_data['labels'][ind].shape[0]

        rps = torch.randint(0, num_labels-1, (args.beam_size, num_segments))
        
        beams.append(rps[:args.beam_size].numpy())        
    
    return beams


def calc_liks(beams, lik_map, ind):
    ll = torch.zeros(beams.shape[0])

    p_info = {}
    for name, lik in lik_map.items():
        bll, bp_info = lik.calc_lik(beams, ind)
        ll += bll
        if bp_info is not None:
            p_info[name] = bp_info
        
            
    return ll, p_info
              

def search_map(args, ind, beams, lik_map):
        
    beam_liks, p_info = calc_liks(beams, lik_map, ind)        
    best_la = beams[beam_liks.argmax().item()]
    
    return best_la
    
def run_map(args):

    random.seed(args.rd_seed)
    np.random.seed(args.rd_seed)
    torch.manual_seed(args.rd_seed)
    
    grammar = Grammar(args.category)
    args.outpath = 'lik_mods/model_output'
    args.grammar = grammar
    
    lik_map = {
        'sem_lik': SemLik(args, grammar),
        'reg_lik': RegLik(args, grammar),    
    }
    
    args.batch_size = args.eval_batch_size
    
    os.system(f'mkdir {args.outpath} > /dev/null 2>&1')    
    os.system(f'mkdir {args.outpath}/{args.exp_name} > /dev/null 2>&1')
    
    test_data = data_utils.load_dataset(
        f'{args.data_path}/{args.category}',
        f'{args.category}/split.json',
        f'{args.set_name}_{args.eval_size}',
    )

    test_data['samps'] = []
    test_data['samp_labels'] = []
    test_data['samp_segments'] = []

    for meshes, labels in \
        tqdm(list(zip(test_data['meshes'], test_data['labels']))):

        seg_labels = grammar.level_map[labels]

        samps, samp_segments, samp_labels = sampleShape(
            meshes,
            seg_labels
        )
        
        test_data['samps'].append(samps.cpu())
        test_data['samp_labels'].append(samp_labels.cpu())
        test_data['samp_segments'].append(samp_segments.cpu())

        
    print("Loading Proposals")

    prop_net = PointNetPPCls(
        len(grammar.terminals),
        1,
        False
    )

    if args.prop_model_path is not None:
        prop_model_path = args.prop_model_path
    else:
        prop_model_path = f'model_output/{args.prop_exp_name}_{args.category}_{args.train_size}/prop/models/prop_net.pt'
        
    prop_net.load_state_dict(torch.load(prop_model_path))        
    prop_net.eval()
    prop_net.to(device)

    if args.guide == 'no':
        utils.log_print("NO GUIDE", args)
        init_beams = loadUniformSamples(test_data, args, grammar)

    else:
        utils.log_print("USING GUIDE", args)        
        _, init_beams = loadPropInfo(prop_net, test_data, args, grammar)

    del prop_net
    
    print("Setting up Liks")
    for lik in lik_map.values():
        lik.load_for_map(args, test_data)
        
    la_preds = []

    print("Starting MAP Search")

    
    for ind in tqdm(list(range(len(init_beams)))):
        map_la = search_map(
            args,
            ind,
            init_beams[ind].astype('long'),
            lik_map,
        )
            
        la_preds.append(map_la)

    res = {}
    res['search_iou'] = calc_metrics(la_preds, test_data, grammar, args)
    
    utils.log_print(f'Results: ', args, 'res')
    for k,v in res.items():
        utils.log_print(f'  {k} : {round(v, 4)}', args, 'res')
    
    
if __name__ == '__main__':
    arg_list = [
        ('-ebs', '--eval_batch_size', 100, int),
        ('-bs', '--beam_size', 10000, int),        
        ('-sen', '--set_name', 'test', str),
        ('-len', '--load_exp_name', 'ngsp_lik', str),
        ('-pen', '--prop_exp_name', 'ngsp_guide', str),
        ('-pmp', '--prop_model_path', None, str),
        ('-guide', '--guide', 'neural', str),
    ] + utils.SEARCH_DEF_ARGS
        
    args = utils.getArgs(arg_list)
    
    assert args.load_exp_name is not None
    
    with torch.no_grad():
        run_map(args)
