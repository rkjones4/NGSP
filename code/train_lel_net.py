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
from models import LelPointNetPPSeg
from lel_net.ss_loss import get_selfsup_loss
from focal_loss import FocalLoss

MODEL_TYPE = 'lel'

ss_loss = get_selfsup_loss()
celoss = FocalLoss(alpha=0.5, gamma = 2., reduction='none')        

LEL_TRAIN_LOG_INFO = [
    ('Loss', 'loss', 'count'),
    ('CE Loss', 'ce_loss', 'count'),
    ('SS Loss', 'ss_loss', 'count'),    
    ('Acc', 'corr', 'total')
]

class Dataset:
    def __init__(
        self, data, unsup_data, grammar, batch_size, num_to_eval = -1,
        scale_weight = 0., noise_weight = 0.
    ):

        self.scale_weight = scale_weight
        self.noise_weight = noise_weight

        self.mode = 'train'
        
        self.shape_inds = data['inds']
        
        self.samps = []
        self.samp_segments = []
        self.samp_labels = []
        self.seg_labels = []
        self.label_weights = []
        
        for i, (meshes, labels) in tqdm(
            enumerate(zip(data['meshes'], data['labels'])),
            total=len(data['labels'])
        ):
            seg_labels = grammar.level_map[labels]
            self.seg_labels.append(seg_labels)
            with torch.no_grad():
                samps, samp_segments, samp_labels = self.sampleShape(
                    meshes,
                    seg_labels
                )
                samps = samps.numpy().astype('float16')
                samp_segments = samp_segments.numpy().astype('int16')
                samp_labels = samp_labels.numpy().astype('int16')

            self.samps.append(samps)
            self.samp_segments.append(samp_segments)
            self.samp_labels.append(samp_labels)
            self.label_weights.append(torch.ones(samp_labels.shape))
            
        self.batch_size = batch_size
        self.grammar = grammar

        self.num_to_eval = min(num_to_eval, len(self.shape_inds)) \
                           if num_to_eval > 0 else len(self.shape_inds)

        if unsup_data is None:
            return
                                            
        for i, meshes in tqdm(
            enumerate(unsup_data['meshes']),
            total=len(unsup_data['meshes'])
        ):

            with torch.no_grad():
                samps, samp_segments, samp_labels = self.sampleShape(
                    meshes,
                    torch.zeros(len(meshes))
                )
                samps = samps.numpy().astype('float16')
                samp_segments = samp_segments.numpy().astype('int16')
                samp_labels = samp_labels.numpy().astype('int16')
                
            self.samps.append(samps)
            self.samp_segments.append(samp_segments)
            self.samp_labels.append(samp_labels)
            self.label_weights.append(torch.zeros(samp_labels.shape))
        

        
    def __iter__(self):

        if self.mode == 'train':
            yield from self.train_iter()

        elif self.mode == 'eval':
            yield from self.eval_iter()

        else:
            assert False, f'bad mode {self.mode}'

    def train_iter(self):

        inds = list(range(len(self.samps)))
        random.shuffle(inds)
        
        while len(inds) > 0:                        
            with torch.no_grad():
                batch_shapes = []
                batch_labels = []
                batch_segments = []
                batch_label_weights = []
                
                for _ in range(0, self.batch_size):
                    if len(inds) == 0:
                        break

                    ind = inds.pop(0)
                                        
                    samps = torch.from_numpy(self.samps[ind]).float().to(device)
                    samp_labels = torch.from_numpy(self.samp_labels[ind]).long().to(device)
                    samp_segments = torch.from_numpy(self.samp_segments[ind]).long().to(device)
                    
                    batch_shapes.append(samps)
                    batch_labels.append(samp_labels)
                    batch_segments.append(samp_segments)
                    batch_label_weights.append(
                        self.label_weights[ind].float().to(device)
                    )
                                        
                b_shapes = torch.stack(batch_shapes)
                b_labels = torch.stack(batch_labels)
                b_segments = torch.stack(batch_segments)
                b_label_weights = torch.stack(batch_label_weights)

                scale = (torch.randn(b_shapes.shape[0], 1, b_shapes.shape[2], device=device) * self.scale_weight) + 1.
                noise = torch.randn(b_shapes.shape, device=device) * self.noise_weight

                assert b_shapes.shape[2] == 3
                
                a_shapes = (b_shapes * scale) + noise    
                
            yield a_shapes, b_labels, b_segments, b_label_weights
    
    def eval_iter(self):
        
        inds = list(range(self.num_to_eval))
        
        while len(inds) > 0:                        
            with torch.no_grad():

                ind = inds.pop(0)
                
                samps = torch.from_numpy(self.samps[ind]).float()

                samp_labels = torch.from_numpy(
                    self.samp_labels[ind]
                ).long()

                samp_segments = torch.from_numpy(
                    self.samp_segments[ind]
                ).long()
                
                seg_labels = torch.from_numpy(self.seg_labels[ind]).long()
                
            yield samps.unsqueeze(0), \
                samp_labels.unsqueeze(0), \
                samp_segments.unsqueeze(0), \
                seg_labels.unsqueeze(0), \
                [self.shape_inds[ind]]
            
    def sampleShape(self, meshes, seg_labels):
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
        
        samps, face_inds, _ = utils.sample_surface(
            faces, verts.unsqueeze(0), utils.NUM_PROP_INPUT_PTS
        )
        samps = samps.cpu()
        face_inds.cpu()
        
        samp_segments = segments[face_inds]    
        samp_labels = labels[face_inds]
        
        return samps, samp_segments, samp_labels
    
def model_eval_group_seg(eval_inp):

    net = eval_inp['net']
    points = eval_inp['points'].unsqueeze(0)
    segments = eval_inp['segments']
    
    preds = net(points.to(device))[0][0]

    labels = torch.zeros(points.shape[1]).long().to(device) - 1

    for i in segments.unique():
        seg_inds = (segments == i).nonzero().squeeze().view(-1)        
        seg_preds = preds[seg_inds]
        seg_label = torch.softmax(seg_preds, dim=1).mean(dim=0).argmax().item()
        labels[seg_inds] = seg_label

    assert (labels >= 0).all(), 'some -1 label left'

    return labels.cpu(), None

def model_eval_regions(eval_inp):

    net = eval_inp['net']
    points = eval_inp['points'].unsqueeze(0)
    segments = eval_inp['segments']
    num_segments = eval_inp['num_segments']
    
    preds = net(points.to(device))[0][0]
    
    labels = torch.zeros(num_segments).long()

    for i in range(num_segments):
        seg_inds = (segments == i).nonzero().flatten()
        seg_preds = preds[seg_inds]
        seg_label = torch.softmax(seg_preds, dim=1).mean(dim=0).argmax().item()
        labels[i] = seg_label
        
    return labels

def model_eval_no_group(eval_inp):

    net = eval_inp['net']
    points = eval_inp['points'].unsqueeze(0)
        
    probs, _ = net(points.to(device))
    preds= probs[0].argmax(dim=1)
    
    return preds.cpu(), None


def model_train_batch(batch, net, opt):
    samps, labels, segments, label_weights = batch
    
    br = {}

    preds, emb_out = net(samps)
        
    c_loss = celoss(
        preds.reshape(-1, preds.shape[2]),
        labels.flatten()
    )

    c_loss = (c_loss * label_weights.flatten()).sum() / (label_weights.sum() + 1e-8)

    s_loss = ss_loss(emb_out[:,:,:net.ss_num_points], segments[:,:net.ss_num_points])

    loss = c_loss + net.lam * s_loss
    
    with torch.no_grad():
        corr = 1e-8 + ((
            preds.reshape(-1, preds.shape[2]).argmax(dim=1) \
            == labels.flatten()
        ) * label_weights.flatten()).sum().item() * 1.
        total = 1e-8 + label_weights.sum().item()

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    br['loss'] = loss.item()
    br['ss_loss'] = s_loss.item()
    br['ce_loss'] = c_loss.item()    
    br['corr'] = corr
    br['total'] = total
    br['count'] = 1.
    return br

def eval_only(args):

    utils.init_model_run(args)

    grammar = Grammar(args.category)
    
        
    test_data = data_utils.load_dataset(
        f'{args.data_path}/{args.category}',
        f'{args.category}/split.json',
        f'test_{args.eval_size}',
    )

    test_loader = Dataset(
        test_data,
        None,
        grammar,
        args.batch_size,        
    )

    net = LelPointNetPPSeg(
        len(grammar.terminals),
        0,
        args.emb_dim,
        USE_BN=args.use_bn,
        DP=args.dropout
    )

    net.lam = args.lam
    net.ss_num_points = args.ss_num_points
    
    net.load_state_dict(
        torch.load(
            args.eval_model_path
        )
    )
    net.eval()
    net.to(device)

    test_loader.mode = 'eval'

    group_res = eval_utils.model_eval(
        args,
        test_loader,
        net,
        0,
        grammar,
        'blank',
        model_eval_group_seg
    )

    utils.log_print(f"~~ TEST EVAL Result for Grouping ~~", args)

    utils.print_results(
        utils.EVAL_PROP_LOG_INFO,
        group_res,
        args
    )

    
def main(args):

    if args.eval_only == 'y':
        return eval_only(args)
    
    utils.init_model_run(args, MODEL_TYPE)
    
    grammar = Grammar(args.category)

    unsup_data = data_utils.load_unsup_dataset(
        f'{args.data_path}/{args.category}',
        f'{args.category}/split.json',
        args.train_size,
        args.unsup_size
    )
    
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
    
    print(f"Num training data {len(train_data['inds'])}")
    print(f"Num val data {len(val_data['inds'])}")
    print(f"Num test data {len(test_data['inds'])}")
        
    train_loader = Dataset(
        train_data,
        unsup_data,
        grammar,
        args.batch_size,
        num_to_eval=len(test_data['inds']),
        scale_weight = args.scale_weight,
        noise_weight = args.noise_weight,
    )

    val_loader = Dataset(
        val_data,
        None,
        grammar,
        args.batch_size,        
    )

    test_loader = Dataset(
        test_data,
        None,
        grammar,
        args.batch_size,        
    )

    net = LelPointNetPPSeg(
        len(grammar.terminals),
        0,
        args.emb_dim,
        USE_BN=args.use_bn,
        DP=args.dropout
    )

    net.lam = args.lam
    net.ss_num_points = args.ss_num_points
    
    net.to(device)
    opt = torch.optim.Adam(
        net.parameters(),
        lr = args.lr,
        betas=(0.5, 0.99)
    )
    
    res = {
        'train_plots': {'train':{}, 'val':{}},
        'train_epochs': [],
        'eval_plots': {'train':{}, 'val':{}, 'test':{}},
        'eval_epochs': []
    }

    save_model_weights = {}
    
    eval_data = [
        ('train', train_loader),
        ('val', val_loader),
        ('test', test_loader)
    ]

    for e in range(args.epochs):

        train_loader.mode = 'train'
        val_loader.mode = 'train'
        
        train_utils.run_train_epoch(
            args,
            res,
            net,
            opt,
            train_loader,
            val_loader,
            LEL_TRAIN_LOG_INFO,
            e,
            model_train_batch
        )

        train_loader.mode = 'eval'
        val_loader.mode = 'eval'
        test_loader.mode = 'eval'
                                    
        best_ep = eval_utils.run_eval_epoch(
            args,
            res,
            net,
            eval_data,
            utils.EVAL_PROP_LOG_INFO,
            e,
            grammar,
            model_eval_group_seg
        )
        
        if best_ep > 0:
            break

        if (e+1) % args.eval_per == 0:
            save_model_weights[e] = deepcopy(net.state_dict())
            
    torch.save(
        save_model_weights[best_ep],
        f"{args.outpath}/{args.exp_name}/models/{MODEL_TYPE}_net.pt"
    )

    utils.log_print("Saving Best Model", args)    
    net.load_state_dict(
        save_model_weights[best_ep]
    )
    net.eval()

    test_loader.mode = 'eval'
    with torch.no_grad():
        no_group_res = eval_utils.model_eval(
            args,
            test_loader,
            net,
            0,
            grammar,
            'blank',
            model_eval_no_group
        )

        group_res = eval_utils.model_eval(
            args,
            test_loader,
            net,
            0,
            grammar,
            'blank',
            model_eval_group_seg
        )

        utils.log_print(f"~~ Final Test Set Result for No Grouping ~~", args)

        utils.print_results(
            utils.EVAL_PROP_LOG_INFO,
            no_group_res,
            args
        )

        utils.log_print(f"~~ Final Test Set Result for Grouping ~~", args)

        utils.print_results(
            utils.EVAL_PROP_LOG_INFO,
            group_res,
            args
        )
    
            
if __name__ == '__main__':
    arg_list = [
        ('-drop', '--dropout', 0.4, float),
        ('-ns', '--num_samples', 10000, int),
        ('-scalew', '--scale_weight', 0.2, float),
        ('-noisew', '--noise_weight', 0.02, float),
        ('-fl', '--focal_loss', 'True', str),
        ('-ed', '--emb_dim', 128, int),
        ('-lam', '--lam', 10, float),
        ('-ssnp', '--ss_num_points', 1024, int),        
        ('-us', '--unsup_size', None,  int),        
    ]        
    args = utils.getArgs(arg_list)
    main(args)
    
