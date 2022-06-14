from grammar import Grammar
import sys, os, torch
import numpy as np
import argparse
import data_utils
import utils
from torch.utils.data import DataLoader
import train_utils, eval_utils
from models import PointNetPPCls
from utils import device
import ast
from copy import deepcopy, copy
import random
from tqdm import tqdm
from focal_loss import FocalLoss
celoss = None

MODEL_TYPE = 'prop'

class Dataset:
    def __init__(
        self, data, grammar, batch_size, num_to_eval = -1, scale_weight=0., noise_weight= 0.,
    ):
        self.scale_weight = scale_weight
        self.noise_weight = noise_weight
        
        self.mode = 'train'
        
        self.shape_inds = data['inds']
        
        self.samps = []
        self.samp_segments = []
        self.samp_labels = []
        self.seg_labels = []
        
        self.flat_samps = []
        self.flat_labels = []
                
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
                n_samps = samps.numpy().astype('float16')
                n_samp_segments = samp_segments.numpy().astype('int16')
                n_samp_labels = samp_labels.numpy().astype('int16')

                self.samps.append(n_samps)
                self.samp_segments.append(n_samp_segments)
                self.samp_labels.append(n_samp_labels)
            
                for j, (_,sl) in enumerate(zip(meshes, seg_labels)):

                    seg_inds = (samp_segments == j).nonzero().flatten()
                    seg_onehot = torch.zeros(samps.shape[0], 1)
                    seg_onehot[seg_inds] = 1.0

                    shape = torch.cat((samps, seg_onehot), dim=1).numpy().astype('float16')

                    self.flat_samps.append(shape)
                    self.flat_labels.append(sl.item())

        self.flat_samps = np.stack(self.flat_samps)
        self.flat_labels = np.array(self.flat_labels).astype('int16')
        
        self.batch_size = batch_size
        self.grammar = grammar

        self.num_to_eval = min(num_to_eval, len(self.shape_inds)) \
                           if num_to_eval > 0 else len(self.shape_inds)



        
    def __iter__(self):

        if self.mode == 'train':
            yield from self.train_iter()
            
        elif self.mode == 'eval':
            yield from self.eval_iter()

        else:
            assert False, f'bad mode {self.mode}'

    def train_iter(self):

        inds = torch.randperm(self.flat_samps.shape[0]).numpy()

        for i in range(0, self.flat_samps.shape[0], self.batch_size):        
            with torch.no_grad():

                binds = inds[i:i+self.batch_size]

                
                b_shapes = torch.from_numpy(self.flat_samps[binds]).float().to(device)
                b_labels = torch.from_numpy(self.flat_labels[binds]).long().to(device)
                            
                scale = (torch.randn(b_shapes.shape[0], 1, b_shapes.shape[2], device=device) * self.scale_weight) + 1.
                noise = torch.randn(b_shapes.shape, device=device) * self.noise_weight
                                
                scale[:,:,3:] = 1.
                noise[:,:,3:] = 0.

                a_shapes = (b_shapes * scale) + noise    
                
            yield a_shapes, b_labels

    
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
        
        samps, face_inds, normals = utils.sample_surface(
            faces, verts.unsqueeze(0), utils.NUM_PROP_INPUT_PTS
        )
            
        samps = samps.cpu()
        face_inds.cpu()
        
        samp_segments = segments[face_inds]    
        samp_labels = labels[face_inds]
        
        return samps, samp_segments, samp_labels                                
        

def model_eval(eval_inp):
    net = eval_inp['net']
    
    points = eval_inp['points']
    segments = eval_inp['segments']    
    num_segments = eval_inp['num_segments']
    batch_size = eval_inp['batch_size']

    mode = eval_inp['mode']
                
    seg_label_probs = []

    seen = torch.zeros(points.shape[0]).long().to(device)
    uni_segs = set(segments.cpu().unique().tolist())

    shape_inds = []
    shapes = []
    samp_inds_array = []
    
    for i in range(num_segments):

        if i not in uni_segs:
            fake = torch.zeros(net.num_classes).to(device)
            fake[0] = 1.0
            seg_label_probs.append(fake)
            continue
        
        seg_inds = (segments == i).nonzero().squeeze().view(-1)
        seen[seg_inds] = 1.0
        
        onehot = torch.zeros(points.shape[0], 1,device=points.device).float()
        onehot[seg_inds] = 1.0        
        shape = torch.cat((points, onehot), dim = 1).to(device)

        seg_label_probs.append(None)
        shapes.append(shape)
        shape_inds.append(i)
        samp_inds_array.append(seg_inds)
        
    assert (seen > 0).all(), 'some -1 label left'
        
    shapes = torch.stack(shapes)
    seg_dist = []

    for i in range(0, shapes.shape[0], batch_size):
        seg_dist.append(torch.softmax(net(shapes[i:i+batch_size]), dim = 1))

    seg_dist = torch.cat(seg_dist, dim=0)
        
    for si, sd in zip(shape_inds, seg_dist):
        seg_label_probs[si] = sd

    if mode == 'sample':
        num_samples = eval_inp['num_samples']
        keep_ll = eval_inp['keep_ll']
        
        samples = eval_utils.search_beam(seg_label_probs, num_samples, keep_ll)

        if len(samples) < num_samples:

            if keep_ll:
            
                samples = samples + [
                    (torch.tensor(-1000.).float(), torch.zeros(samples[0][1].shape).long()) \
                    for _ in range(num_samples - len(samples))
                ]

            else:

                samples = samples + [
                    torch.zeros(samples[0].shape).long() \
                    for _ in range(num_samples - len(samples))
                ]
        
        if 'add_gt' in eval_inp:
            seg_labels = eval_inp['seg_labels']

            sl_ll= 0.

            for i,sl in enumerate(seg_labels):
                sl_ll += torch.log(seg_label_probs[i][sl])

            samples.append((sl_ll.cpu(), seg_labels.cpu()))            

        if 'keep_dist' in eval_inp and eval_inp['keep_dist'] is True:
            return samples, torch.stack(seg_label_probs, dim=0).cpu()
        else:
            return samples

    elif mode == 'point':
    
        labels = torch.zeros(segments.shape).long().to(device) - 1
            
        for seg_ind, samp_inds in zip(shape_inds, samp_inds_array):            
            dist = seg_label_probs[seg_ind]
            pred = dist.argmax()
            labels[samp_inds] = pred

        assert (labels >= 0).all(), 'some -1 label left'
        
        return labels.cpu(), torch.stack(seg_label_probs, dim=0).cpu()
        

    
    
def model_train_batch(batch, net, opt):
    samps, labels = batch
    
    br = {}

    preds = net(samps)

    loss = celoss(
        preds,
        labels
    )
    
    with torch.no_grad():
        corr = (preds.argmax(dim=1) == labels).sum().item() * 1.
        total = labels.shape[0] * 1.

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    br['loss'] = loss.item()
    br['corr'] = corr
    br['total'] = total
    
    return br
    
def main(args):

    utils.init_model_run(args, MODEL_TYPE)
    
    grammar = Grammar(args.category)

    
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
        grammar,
        args.batch_size,
        num_to_eval=len(test_data['inds']),
        scale_weight = args.scale_weight,
        noise_weight = args.noise_weight,
    )

    val_loader = Dataset(
        val_data,
        grammar,
        args.batch_size,
    )

    test_loader = Dataset(
        test_data,
        grammar,
        args.batch_size,
    )


    net = PointNetPPCls(
        len(grammar.terminals),
        1,
        USE_BN=args.use_bn,
        DP=args.dropout
    )
        
    net.to(device)
    
    opt = torch.optim.Adam(
        net.parameters(),
        lr = args.lr,
        eps = 1e-6
    )
    
    global celoss

    if args.focal_loss:
        celoss = FocalLoss(alpha=0.5, gamma = 2., reduction='mean')        
    else:
        celoss = torch.nn.CrossEntropyLoss()
    
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
            utils.PROP_TRAIN_LOG_INFO,
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
            model_eval
        )
        
        if best_ep >= 0:
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

    utils.log_print("Saving Proposals", args)    

    train_loader.num_to_eval = len(train_data['inds'])
    
    for key, loader in eval_data:
        utils.log_print(f" ~~ Final Results For {key} ~~", args)
        with torch.no_grad():
            eval_utils.model_eval_samples(
                args,
                net,
                loader,
                utils.EVAL_SAMPLE_LOG_INFO,
                grammar,
                model_eval
            )
        
if __name__ == '__main__':
    arg_list = [
        ('-drop', '--dropout', 0.4, float),
        ('-ns', '--num_samples', 10000, int),
        ('-scalew', '--scale_weight', 0.2, float),
        ('-noisew', '--noise_weight', 0.02, float),
        ('-fl', '--focal_loss', 'True', str),
    ]    
    args = utils.getArgs(arg_list)
    args.focal_loss = ast.literal_eval(args.focal_loss)
    main(args)
