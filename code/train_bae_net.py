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

from bae_net.bae_net import BaeNet
import bae_net.gen_data as gd

MODEL_TYPE = 'bae'

bceloss = torch.nn.BCELoss()
celoss = torch.nn.CrossEntropyLoss()

BAE_TRAIN_LOG_INFO = [
    ('Guide Loss', 'guide_loss', 'guide_count'),
    ('Unsup Loss', 'unsup_loss', 'unsup_count'),
    ('Guide Acc', 'corr', 'total')
]
guide_count = 0

class Dataset:
    def __init__(
        self, data, unsup_data, grammar, batch_size, num_to_eval = -1, 
    ):

        self.mode = 'train'
        
        self.shape_inds = data['inds']
        
        self.samps = []
        self.samp_segments = []
        self.samp_labels = []
        self.seg_labels = []

        self.voxels = []
        self.occ_pts = []
        self.occ_tars = []
        
        for i, (ind, meshes, labels) in tqdm(
            enumerate(zip(data['inds'], data['meshes'], data['labels'])),
            total=len(data['labels'])
        ):
                        
            seg_labels = grammar.level_map[labels]
            self.seg_labels.append(seg_labels)
            with torch.no_grad():
                samps, samp_segments, samp_labels, shape_verts, shape_faces  = self.sampleShape(
                    meshes,
                    seg_labels
                )
                samps = samps.numpy().astype('float16')
                samp_segments = samp_segments.numpy().astype('int16')
                samp_labels = samp_labels.numpy().astype('int16')

            occ_pts, occ_tars, voxels = gd.load_baenet_data(ind, shape_verts, shape_faces)
            
            self.voxels.append(voxels)
            self.occ_pts.append(occ_pts)
            self.occ_tars.append(occ_tars)
                
            self.samps.append(samps)
            self.samp_segments.append(samp_segments)
            self.samp_labels.append(samp_labels)
                                                                    
        self.batch_size = batch_size
        self.grammar = grammar

        self.num_to_eval = min(num_to_eval, len(self.shape_inds)) \
                           if num_to_eval > 0 else len(self.shape_inds)


        self.unsup_shape_inds = []
        self.unsup_voxels = []
        self.unsup_occ_pts = []
        self.unsup_occ_tars = []
        
        if unsup_data is None:
            return

        self.guide_period = unsup_data['guide_period']
        
        for i, (ind, meshes) in tqdm(
            enumerate(zip(unsup_data['inds'], unsup_data['meshes'])),
            total=len(data['meshes'])
        ):
            
            with torch.no_grad():
                _,_,_, shape_verts, shape_faces  = self.sampleShape(
                    meshes,
                    [-1 for _ in range(len(meshes))]
                )

            occ_pts, occ_tars, voxels = gd.load_baenet_data(ind, shape_verts, shape_faces)
                   
            self.unsup_voxels.append(voxels)
            self.unsup_occ_pts.append(occ_pts)
            self.unsup_occ_tars.append(occ_tars)
            self.unsup_shape_inds.append(ind)
        
    def __iter__(self):

        if self.mode == 'warmup':
            yield from self.guide_iter()

        elif self.mode == 'train':
            yield from self.unsup_iter()

        elif self.mode == 'eval':
            yield from self.eval_iter()

        else:
            assert False, f'bad mode {self.mode}'

    def guide_iter(self):

        inds = list(range(len(self.shape_inds)))
        random.shuffle(inds)
        
        while len(inds) > 0:                        
            with torch.no_grad():
                ind = inds.pop(0)

                voxels = torch.from_numpy(self.voxels[ind]).float().to(device)
                occ_pts = torch.from_numpy(self.occ_pts[ind]).float().to(device)
                occ_tars = torch.from_numpy(self.occ_tars[ind]).float().to(device)
                
                samps = torch.from_numpy(self.samps[ind]).float().to(device)
                samp_labels = torch.from_numpy(self.samp_labels[ind]).long().to(device)
                
            yield voxels, occ_pts, occ_tars, samps, samp_labels
            
                
    def unsup_iter(self):
        global guide_count
        inds = list(range(len(self.unsup_shape_inds)))
        random.shuffle(inds)    
        
        while len(inds) > 0:
            if guide_count == self.guide_period:
                guide_count = 0
                yield from self.guide_iter()
                
            with torch.no_grad():
                ind = inds.pop(0)

                voxels = torch.from_numpy(self.unsup_voxels[ind]).float().to(device)
                occ_pts = torch.from_numpy(self.unsup_occ_pts[ind]).float().to(device)
                occ_tars = torch.from_numpy(self.unsup_occ_tars[ind]).float().to(device)            

            guide_count += 1
            yield voxels, occ_pts, occ_tars, None, None
        
    
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
        
        return samps, samp_segments, samp_labels, verts.cpu().numpy(), faces.cpu().numpy()
    
def model_eval_group_seg(eval_inp):
    
    net = eval_inp['net']
    points = eval_inp['points'].to(device)
    segments = eval_inp['segments']    

    ind = eval_inp['ind']
    
    _,_,voxels = gd.load_baenet_data(ind,None,None)

    code = net.encode(torch.from_numpy(voxels).float().to(device))
    
    _, preds = net(
        torch.cat((
            code.view(1, -1).repeat(points.shape[0], 1), points
        ), dim = 1)
    )
    
    labels = torch.zeros(points.shape[0]).long().to(device) - 1

    for i in segments.unique():
        seg_inds = (segments == i).nonzero().squeeze().view(-1)        
        seg_preds = preds[seg_inds]
        seg_label = torch.softmax(seg_preds, dim=1).mean(dim=0).argmax().item()
        labels[seg_inds] = seg_label

    assert (labels >= 0).all(), 'some -1 label left'

    return labels.cpu(), None


def model_eval_regions(eval_inp):

    net = eval_inp['net']
    points = eval_inp['points'].to(device)
    segments = eval_inp['segments']
    num_segments = eval_inp['num_segments']
    
    ind = eval_inp['ind']
    
    _,_,voxels = gd.load_baenet_data(ind,None,None)

    code = net.encode(torch.from_numpy(voxels).float().to(device))
    
    _, preds = net(
        torch.cat((
            code.view(1, -1).repeat(points.shape[0], 1), points
        ), dim = 1)
    )
    
    labels = torch.zeros(num_segments).long()

    for i in range(num_segments):
        seg_inds = (segments == i).nonzero().flatten()
        seg_preds = preds[seg_inds]
        seg_label = torch.softmax(seg_preds, dim=1).mean(dim=0).argmax().item()
        labels[i] = seg_label
        
    return labels


def model_train_batch(batch, net, opt):
    voxels, points, targets, samps, labels = batch

    if samps is None and labels is None:
        return unsup_model_train_batch(voxels, points, targets, net, opt)

    else:
        return guide_model_train_batch(voxels, points, targets, samps, labels, net, opt)

def unsup_model_train_batch(voxels, points, targets, net, opt):

    code = net.encode(voxels)
                        
    shape, _ = net(
        torch.cat((
            code.view(1, -1).repeat(points.shape[0], 1), points
        ), dim = 1)
    )

    l3_params = torch.cat([x.view(-1) for x in net.net.l3.parameters()])
    l1reg_loss = net.lreg * torch.norm(l3_params, 1)
    
    loss = ((shape[:,0] - targets[:,0]) **2).mean() + l1reg_loss
                                        
    opt.zero_grad()
    loss.backward()            
    opt.step()

    br = {}
    
    br['unsup_loss'] = loss.item()
    br['unsup_count'] = 1
    
    return br
    
def guide_model_train_batch(voxels, points, targets, samps, labels, net, opt):
    
    code = net.encode(voxels)
    
    recon_loss = 0.

    shape, _ = net(
        torch.cat((
            code.view(1, -1).repeat(points.shape[0], 1), points
        ), dim = 1)
    )
    
    recon_loss = ((shape[:,0] - targets[:,0]) **2).mean() 
        
    _, l3 = net(
        torch.cat((
            code.view(1, -1).repeat(samps.shape[0], 1), samps
        ), dim = 1)
    )
        
    part_loss = 0.
        
    for i in range(l3.shape[1]):
        target = (labels == i).float()
        pred = l3[:,i]
        part_loss += ((pred - target) ** 2).mean()

    recon_loss += (part_loss / (l3.shape[1] * 1.0))


    l3_params = torch.cat([x.view(-1) for x in net.net.l3.parameters()])
    l1reg_loss = net.lreg * torch.norm(l3_params, 1)
    
    loss = recon_loss + l1reg_loss
    
    br = {}

    with torch.no_grad():
        corr = (l3.argmax(dim=1) == labels).sum().item() * 1.
        total = labels.shape[0]

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    br['guide_loss'] = loss.item()
    br['guide_count'] = 1
    br['corr'] = corr
    br['total'] = total
    
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

    net = BaeNet(len(grammar.terminals),)
    net.lreg = args.lreg
    
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

    # Load all shapes not in train/val/test
    unsup_data = data_utils.load_unsup_dataset(
        f'{args.data_path}/{args.category}',
        f'{args.category}/split.json',
        args.train_size,
        args.unsup_size
    )
    unsup_data['guide_period'] = args.guide_period
    
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
            

    net = BaeNet(len(grammar.terminals),)
    net.lreg = args.lreg
    
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

    val_loader.mode = 'eval'
    test_loader.mode = 'eval'
    
    for e in range(-1 * args.warmup, args.epochs):
        
        if e < 0:
            train_loader.mode = 'warmup'        
        else:
            train_loader.mode = 'train'
            
        train_utils.run_train_epoch(
            args,
            deepcopy(res),
            net,
            opt,
            train_loader,
            None,
            BAE_TRAIN_LOG_INFO,
            e,
            model_train_batch
        )

        if e < 0:
            continue
        
        train_loader.mode = 'eval'
                                            
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
            
if __name__ == '__main__':
    arg_list = [
        ('-b', '--batch_size', 1,  int),
        ('-lr', '--lr', 1e-4,  float),
        ('-w', '--warmup', 3000, int),
        ('-gp', '--guide_period', 4, int),
        ('-lreg', '--lreg', 0.000001, float),
        ('-ns', '--num_samples', 0.000001, float),
        ('-evp', '--eval_per', 1,  int),    
        ('-esp', '--es_patience', 10,  int),
        ('-us', '--unsup_size', None,  int),
    ]
    
    args = utils.getArgs(arg_list)
    main(args)
    
