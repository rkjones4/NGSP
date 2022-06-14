from grammar import Grammar
import sys, os, torch
import numpy as np
import argparse
import sem_label_data_utils as search_data_utils
import data_utils
import utils
from torch.utils.data import DataLoader
import train_utils
from models import PointNetPPCls
from utils import device
import ast
import random
from copy import deepcopy
import eval_utils
from tqdm import tqdm
import time
import pickle

MODEL_TYPE = None

bceloss = torch.nn.BCEWithLogitsLoss()

class Dataset:
    def __init__(
        self, data, grammar, batch_size, scale_weight, noise_weight,
    ):

        self.data = data
        self.mode = 'train'                        
        self.batch_size = batch_size
                        
        self.scale_weight = scale_weight
        self.noise_weight = noise_weight
        
        self.grammar = grammar
        
        self.samps = []
        self.segments = []
        self.seg_maps = []        

        self.eval_samps = []
        self.eval_segments = []
        self.eval_labels = []
        
        with torch.no_grad():
            for meshes, seg_labels in tqdm(list(zip(self.data['meshes'], self.data['labels']))):
                seg_labels = torch.from_numpy(grammar.level_map[seg_labels]).long().to(device)
                samps, segments, samp_labels = sampleShape(meshes, seg_labels) 
                
                self.eval_samps.append(                    
                    samps[:utils.NUM_EVAL_POINTS].cpu().numpy().astype('float32')
                )
                self.eval_segments.append(segments[:utils.NUM_EVAL_POINTS].cpu().numpy().astype('int16'))
                self.eval_labels.append(samp_labels[:utils.NUM_EVAL_POINTS].cpu().numpy().astype('int16'))
                                
                seg_map = []
                for i in range(len(meshes)):
                    si = (segments == i).nonzero().flatten()[:utils.NUM_SEARCH_INPUT_PTS].long().cpu()
                    seg_map.append(si)

                exp_inds = torch.zeros(segments.shape)
                seg_inds = torch.cat(seg_map,dim=0)
                exp_inds[seg_inds] = 1
                small_inds = exp_inds.nonzero().flatten()
                
                small_samps = samps[small_inds].to(device)
                small_segments = segments[small_inds].to(device)
                
                self.samps.append(small_samps.cpu().numpy().astype('float16'))
                self.segments.append(small_segments.cpu().numpy().astype('int16'))

                s2i_map = {}
                for i in range(len(meshes)):
                    si = (small_segments == i).nonzero().flatten()
                    if si.shape[0] == 0:
                        continue
                    si = si.repeat((utils.NUM_SEARCH_INPUT_PTS//si.shape[0])+1)
                    si = si[:utils.NUM_SEARCH_INPUT_PTS].long()                    
                    s2i_map[i] = si.to(device)
                    
                self.seg_maps.append(s2i_map)
                

    def add_node_data(self, node_data, node, args):

        self.node = node
        self.args = args

        if MODEL_TYPE ==  'geom':
            self.dist = [0.15, 0.15, 0.15, 0.0, 0.05, 0.5]
            
        elif MODEL_TYPE == 'lay':
            self.dist = [0.075,0.075,0.075, 0.25, 0.025, 0.5]
            
        self.hn_prop_inds = node_data['hn_prop_inds']
        self.hn_prop_in_parts = node_data['hn_prop_in_parts']
        self.hn_prop_in_cls = node_data['hn_prop_in_cls']
        
        self.pos_inds = node_data['pos_inds']
        
        self.pos_info = []
        for ip, ic in zip(node_data['pos_in_parts'], node_data['pos_in_cls']):
            self.pos_info.append((ip, ic))

        prop_inds = node_data['prop_inds']
        prop_in_parts = node_data['prop_in_parts']
        prop_in_cls = node_data['prop_in_cls']

        self.pos_prop_inds = node_data['prop_inds']
        self.pos_prop_in_parts = node_data['prop_in_parts']
        self.pos_prop_in_cls = node_data['prop_in_cls']
        
        prop_map = {}
        for i, pi in enumerate(prop_inds):
            if pi not in prop_map:
                prop_map[pi] = []
            prop_map[pi].append(i)

        self.neg_info = []
        self.neg_loss_info = []
        self.neg_match_info = []

        self.areas = node_data['areas']
        
        for pi in range(len(self.pos_inds)):
            
            ind = self.pos_inds[pi]

            areas = self.areas[pi]
                                    
            pos_info = self.pos_info[pi]
            
            if ind not in prop_map:
                prop_map[ind] = []

            prop_negs = prop_map[ind]
            
            neg_info = []
            
            for pn in prop_negs:
                neg_info.append((prop_in_parts[pn], prop_in_cls[pn]))

            num_segments = len(self.data['meshes'][ind])

            num_to_mine = args.min_eval_negs - len(neg_info)

            if num_to_mine > 0:

                if MODEL_TYPE == 'geom':
                    dist = [0.3, 0.3, 0.3, 0.0, 0.0, 0.0]
        
                elif MODEL_TYPE == 'lay':
                    dist = [0.3, 0.3, 0.3, 1.0, 0.0, 0.0]
                
                mined_info, _ = mineNegatives(
                    pos_info,
                    neg_info,
                    num_to_mine,
                    num_segments,
                    self.grammar,
                    node,
                    areas,
                    args,
                    dist,
                    ind,
                    self
                )
                neg_info += mined_info

            self.neg_info.append(neg_info)
            
            
    def __iter__(self):

        if self.mode == 'train':                    
            yield from self.train_iter()            
                
        elif self.mode == 'eval':
            yield from self.eval_iter()

        else:
            assert False, f'bad mode {self.mode}'
            
    def eval_iter(self):

        for pind in range(len(self.pos_inds)):
            ind = self.pos_inds[pind]

            meshes = self.data['meshes'][ind]

            pn_ind = self.data['inds'][ind]
            
            samps = torch.from_numpy(self.samps[ind]).float().to(device)
            segments = torch.from_numpy(self.segments[ind]).long().to(device)
            seg_map = self.seg_maps[ind]
            
            pos_info = self.pos_info[pind]            
            neg_info = self.neg_info[pind]

            if len(neg_info) == 0:
                print('saw 0 neg info')
                continue
            
            yield meshes, samps, segments, seg_map, pos_info, neg_info, pn_ind
            

    def train_iter(self):
        pinds = list(range(len(self.pos_inds))) 
        
        random.shuffle(pinds)
        
        while(len(pinds) > 0):

            pind = pinds.pop(0)

            ind = self.pos_inds[pind]
            pos_info = self.pos_info[pind]

            mined_info, mined_inds = mineNegatives(
                pos_info,
                [],
                self.batch_size,
                len(self.data['meshes'][ind]),
                self.grammar,
                self.node,
                self.areas[pind],
                self.args,
                self.dist,
                ind,
                self
            )
            
            bdata = [(ind, pos_info[0], pos_info[1])]
            blabels = [1.0]

            for (m_ip, m_ic), mi in zip(mined_info, mined_inds):
                bdata.append((mi, m_ip, m_ic))
                blabels.append(0.0)
                
            blabels = torch.tensor(blabels).to(device)
                    
            with torch.no_grad():

                bshapes = []

                for ind, ip, cl in bdata:
                    meshes = self.data['meshes'][ind]
                    samps = torch.from_numpy(self.samps[ind]).float().to(device)
                    segments = torch.from_numpy(self.segments[ind]).long().to(device)
                    seg_map = self.seg_maps[ind]
                    
                    _bshape = make_shapes_base(
                        len(meshes),
                        samps,
                        segments,
                        seg_map,
                        [(ip, cl)],
                        self.grammar,
                        MODEL_TYPE,
                        self.scale_weight,
                        self.noise_weight
                    )
                    bshapes.append(_bshape)

                bshapes = torch.cat(bshapes, dim =0)
                    
            yield bshapes, blabels


def sampOtherNode(num_segments, in_parts, in_cls, node_oh_max, ind, loader):
    out_parts = list(set(range(num_segments)) - set(in_parts.tolist()))

    if len(out_parts) == 0:
        return None
        
    neg_parts = random.sample(out_parts, random.randint(1, len(out_parts)))

    neg_c_labels = np.random.choice(
        node_oh_max,            
        len(neg_parts)
    )
        
    return np.array(neg_parts), neg_c_labels, ind
        
        
def sampAddPart(num_segments, in_parts, in_cls, node_oh_max, ind, loader):
    out_parts = list(set(range(num_segments)) - set(in_parts.tolist()))
        
    if len(out_parts) == 0:
        return None
        
    _neg_parts = random.sample(out_parts, random.randint(1, len(out_parts)))

    neg_parts = in_parts.tolist() + _neg_parts
        
    _neg_c_labels = np.random.choice(
        node_oh_max,
        len(_neg_parts)
    )
    neg_c_labels = in_cls.tolist() + _neg_c_labels.tolist()
                
    return np.array(neg_parts), np.array(neg_c_labels), ind
    
    
def sampRemPart(num_segments, in_parts, in_cls, node_oh_max, ind, loader):
    if len(in_parts) < 2:
        return None

    num_pos = random.randint(1, len(in_parts)-1)
    pinds = random.sample(list(range(in_parts.shape[0])), num_pos)

    neg_parts = [in_parts[pi] for pi in pinds]
    neg_c_labels = [in_cls[pi] for pi in pinds]

    return np.array(neg_parts), np.array(neg_c_labels), ind
        
def sampOtherCls(num_segments, in_parts, in_cls, node_oh_max, ind, loader):
    _clabels = []
    
    for i in range(len(in_parts)):
        if random.random() >= 0.5:
            _clabels.append(
                random.randint(0, node_oh_max-1)
            )
        else:
            _clabels.append(in_cls[i])

    return in_parts, np.array(_clabels), ind

def sampHardNeg(num_segments, in_parts, in_cls, node_oh_max, ind, loader):
    if len(loader.hn_prop_inds) == 0:
        return None
    
    ni = random.randint(0, len(loader.hn_prop_inds)-1)

    ind = loader.hn_prop_inds[ni]
    in_parts = loader.hn_prop_in_parts[ni]
    in_cls = loader.hn_prop_in_cls[ni]
            
    return in_parts, in_cls, ind


def sampProp(num_segments, in_parts, in_cls, node_oh_max, ind, loader):
    if len(loader.pos_prop_inds) == 0:
        return None
    
    pi = random.randint(0, len(loader.pos_prop_inds)-1)

    ind = loader.pos_prop_inds[pi]
    in_parts = loader.pos_prop_in_parts[pi]
    in_cls = loader.pos_prop_in_cls[pi]
            
    return in_parts, in_cls, ind


def getSig(info):
    parts, labels = info
    if MODEL_TYPE == 'lay':        
        z = list(zip(parts, labels))
    elif MODEL_TYPE == 'geom':
        z = deepcopy(parts)            
    z.sort()
    return tuple(z)

def mineNegatives(
    pos_info, neg_info, num_to_mine, num_segments,
    grammar, node, areas, args, dist, ind, loader
):
    if grammar.fi2l[node] in grammar.hi2l:                    
        node_oh_max = min(
            len(grammar.hi2l[grammar.fi2l[node]]),
            grammar.hc_max
        )
    else:
        node_oh_max = grammar.hc_max
        
    smp_fns = {
        0: sampOtherNode,
        1: sampAddPart,
        2: sampRemPart,
        3: sampOtherCls,
        4: sampHardNeg,
        5: sampProp
    }
        
    dist = np.array(dist).astype('float32')
    dist /= dist.sum()

    pos_sig = getSig(pos_info)
    seen_sigs = set([pos_sig])
    
    for info in neg_info:
        seen_sigs.add(getSig(info))

    c = 0

    mined_info = []
    mined_inds = []
    
    in_parts, in_cls = pos_info
    
    while len(mined_info) < num_to_mine:
        if c > 1000:
            break
        
        i = np.random.choice(dist.shape[0], p=dist)
        fn = smp_fns[i]

        mres = fn(
            num_segments, in_parts, in_cls, node_oh_max, ind, loader
        )
        if mres is None:
            c += 1
            continue

        _m0, _m1, mined_ind = mres

        mi = (_m0, _m1)
        
        ms = getSig(mi)

        if ms in seen_sigs and mined_ind == ind:
            c += 1                        
            continue
                
        seen_sigs.add(ms)

        valid_neg = True

        if mined_ind == ind:
        
            valid_neg = search_data_utils.check_valid_neg(
                areas, pos_sig, ms, args, MODEL_TYPE
            )
        
        if valid_neg:
            mined_info.append(mi)
            mined_inds.append(mined_ind)
            
    return mined_info, mined_inds


def sampleShape(meshes, seg_labels):

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
    segments = torch.tensor(segments).long().to(device)
    labels = torch.tensor(labels).long().to(device)
        
    samps, face_inds, _ = utils.sample_surface(
        faces, verts.unsqueeze(0), utils.NUM_SAMP_PTS
    )
        
    samp_segments = segments[face_inds]    
    samp_labels = labels[face_inds]
        
    return samps, samp_segments, samp_labels


def make_shapes(num_segments, samps, segments, seg_map, pos_info, neg_info, grammar, scale_weight, noise_weight):

    all_info = [pos_info] + neg_info

    return make_shapes_base(
        num_segments, samps, segments, seg_map, all_info, grammar, MODEL_TYPE, scale_weight, noise_weight
    )
               

def make_shapes_base(
    num_segments, samps, segments, seg_map, all_info, grammar, name, scale_weight, noise_weight
):

    shapes = []
    
    for in_parts, c_labels in all_info:

        fast_node_inds = torch.zeros(segments.shape, device=device)

        try:
            _fast_inds = [seg_map[i] for i in in_parts if i in seg_map]
            assert len(_fast_inds) > 0, 'zero_samp_segment'
            fast_inds = torch.cat(_fast_inds, dim=0).to(device)        

        except Exception as e:
            if 'zero_samp_segment' in e.args[0]:

                if name == 'lay':
                    shapes.append(torch.zeros(utils.NUM_SEARCH_INPUT_PTS, 3 + grammar.hc_max, device=device).float())
                else:
                    shapes.append(torch.zeros(utils.NUM_SEARCH_INPUT_PTS, 3, device=device).float())
                    
                continue

            else:
                raise e
            
        fast_node_inds[fast_inds] = 1

        node_inds = fast_node_inds.nonzero().flatten()

        if node_inds.shape[0] < utils.NUM_SEARCH_INPUT_PTS:
            node_inds = node_inds.repeat((utils.NUM_SEARCH_INPUT_PTS//node_inds.shape[0])+1)

        node_inds = node_inds[:utils.NUM_SEARCH_INPUT_PTS]        
        node_samps = samps[node_inds]
        
        if name == 'lay':
            c_labels_map = torch.zeros(num_segments).long().to(device) - 1
            
            for ip, cl in zip(in_parts, c_labels):
                c_labels_map[ip] = cl.item()
                        
            node_segs = segments[node_inds]
            node_c_labels = c_labels_map[node_segs]            
            
            assert (node_c_labels >= 0).all(), 'missing some'
                    
            oh_labels = torch.nn.functional.one_hot(node_c_labels, grammar.hc_max).float().to(device)

            node_shape = torch.cat((node_samps, oh_labels), dim=1)
            
        elif name == 'geom':
            node_shape = node_samps
            
        shapes.append(node_shape)
        
    shapes = torch.stack(shapes, dim=0)
    
    c = (shapes.max(dim=1).values + shapes.min(dim=1).values) / 2    
    if name == 'lay':
        c[:,3:] = 0
    c = c.reshape(c.shape[0], 1, c.shape[1])        
    shapes -= c    

    scale = (torch.randn(
        shapes.shape[0], 1, shapes.shape[2],
        device=device) * scale_weight
    ) + 1.
    
    noise = torch.randn(
        shapes.shape,
        device=device
    ) * noise_weight
            
    if name == 'lay':
        scale[:,:,3:] = 1.
        noise[:,:,3:] = 0.

    shapes = (shapes * scale) + noise    
    
    return shapes


def model_eval(inp):

    net = inp['net']
    meshes = inp['meshes']
    samps = inp['samps']
    segments = inp['segments']
    seg_map = inp['seg_map']
    pos_info = inp['pos_info']
    neg_info = inp['neg_info']
    grammar = inp['grammar']
    batch_size = inp['batch_size']
    name = inp['name']
    args = inp['args']
    
    shapes = make_shapes(
        len(meshes), samps, segments, seg_map, pos_info, neg_info, grammar, 0., 0.
    )

    preds = []
    start = 0

    while(start < shapes.shape[0]):
        end = start+batch_size
        preds.append(torch.sigmoid(net(shapes[start:end]).flatten()).cpu())
        start = end
        
    preds = torch.cat(preds, dim=0)

    # dummy value for second arg
    return preds, 0.


def model_train_batch(batch, net, opt):    
    return model_train_batch_hard_neg(batch, net, opt)    
            

def model_train_batch_hard_neg(batch, net, opt):
    br = {}

    shapes, labels = batch

    raw_preds = net(shapes).flatten()

    pos_inds = (labels > 0).nonzero().flatten()
    neg_inds = (labels == 0).nonzero().flatten()

    raw_pos_preds = raw_preds[pos_inds]
    raw_neg_preds = raw_preds[neg_inds]
    
    pos_loss = bceloss(
        raw_pos_preds,
        torch.ones(raw_pos_preds.shape,device=raw_pos_preds.device).float(),
    )
            
    neg_loss = bceloss(
        raw_neg_preds,
        torch.zeros(raw_neg_preds.shape,device=raw_neg_preds.device).float(),
    )
    
    loss = pos_loss + neg_loss
            
    with torch.no_grad():
        
        pos_corr = (raw_pos_preds >= 0.0).sum().item()
        pos_total = raw_pos_preds.shape[0] * 1.
            
        neg_corr = (raw_neg_preds <= 0.0).sum().item()         
        neg_total = raw_neg_preds.shape[0] * 1.

        corr =  pos_corr + neg_corr
        total = pos_total + neg_total

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    br['loss'] = loss.item()
    br['corr'] = corr
    br['total'] = total
    br['pos_corr'] = pos_corr
    br['neg_corr'] = neg_corr
    br['pos_total'] = pos_total
    br['neg_total'] = neg_total
    br['nc'] = 1.

    return br

def getNodeNet(args, grammar, name):

    if name == 'geom':
        num_chans = 0
    elif name == 'lay':
        num_chans = grammar.hc_max


    net = PointNetPPCls(
        1,
        num_chans,
        USE_BN=args.use_bn,
        DP=args.dropout
    )

    return net            

def main(args):    

    os.system(f'mkdir {args.outpath} > /dev/null 2>&1')    
    os.system(f'mkdir {args.outpath}/{args.exp_name} > /dev/null 2>&1')
    
    grammar = Grammar(args.category)
        
    print("Loading Meshes")

    utils.log_print("Making New Samples For Train Search", args)

    
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
                        
    print("Sampling Meshes")
    
    train_loader = Dataset(            
        train_data,
        grammar,
        batch_size = args.batch_size,
        scale_weight=args.scale_weight,
        noise_weight=args.noise_weight,
    )
        
    val_loader = Dataset(
        val_data,
        grammar,
        batch_size = args.batch_size,
        scale_weight=0.,
        noise_weight=0.,
    )
        
    test_loader = Dataset(            
        test_data,
        grammar,
        batch_size = args.batch_size,
        scale_weight=0.,
        noise_weight=0.,
    )
    
    
    main_mt(args, 'geom', train_loader, val_loader, test_loader, grammar)
    main_mt(args, 'lay', train_loader, val_loader, test_loader, grammar)
    
def main_mt(_args, model_type, train_loader, val_loader, test_loader, grammar):

    args = deepcopy(_args)

    global MODEL_TYPE
    MODEL_TYPE = model_type

    utils.init_model_run(args, MODEL_TYPE)
        
    if MODEL_TYPE == 'geom':
        node_list = grammar.get_geom_node_list()

    elif MODEL_TYPE == 'lay':
        node_list = grammar.get_lay_node_list()

    else:
        assert False
        
    res = {}

    if args.node_list is not None:
        node_list = [int(n) for n in args.node_list.split(',')]
        utils.log_print(f"USING NODE LIST : {node_list}", args)

        
    def train_node(n):
        node_args = deepcopy(args)
        node_args.exp_name += f'/node_{grammar.fi2l[n].replace("/","-")}'
        os.system(f'rm {node_args.outpath}/{node_args.exp_name}/ -r > /dev/null 2>&1')
        os.system(f'mkdir {node_args.outpath}/{node_args.exp_name} > /dev/null 2>&1')
        os.system(f'mkdir {node_args.outpath}/{node_args.exp_name}/plots > /dev/null 2>&1')
        os.system(f'mkdir {node_args.outpath}/{node_args.exp_name}/plots/train > /dev/null 2>&1')
        os.system(f'mkdir {node_args.outpath}/{node_args.exp_name}/plots/eval > /dev/null 2>&1')    

        print("Parsing Saved Proposals")
            
        node_train_data = search_data_utils.parseDataWithNode(
            train_loader.data, n, grammar, args, MODEL_TYPE
        )
            
        node_val_data = search_data_utils.parseDataWithNode(
            val_loader.data, n, grammar, args, MODEL_TYPE
        )
        node_test_data = search_data_utils.parseDataWithNode(
            test_loader.data, n, grammar, args, MODEL_TYPE
        )

        num_train = len(node_train_data['pos_inds'])
        num_val = len(node_val_data['pos_inds'])
        num_test = len(node_test_data['pos_inds'])
        
        
        if num_train == 0 or num_val ==0 or num_test == 0:
            utils.log_print(f"SKIPPING {n}:{grammar.fi2l[n]} | T/V/T:({num_train}, {num_val}, {num_test})", args)
            return

        print("Constructing Node Data")

        
        train_loader.add_node_data(
            node_train_data,
            n,
            args,
        )
        val_loader.add_node_data(
            node_val_data,
            n,
            args,
        )
        test_loader.add_node_data(
            node_test_data,
            n,
            args,
        )
        
        utils.log_print(f"Training {n}:{grammar.fi2l[n]} | T/V/T:({num_train}, {num_val}, {num_test})", args)
                    
        net = getNodeNet(args, grammar, MODEL_TYPE)
                
        net.to(device)        
        
        opt = torch.optim.Adam(
            net.parameters(),
            lr = args.lr,
            eps = 1e-6
        )        
        
        cur_res = {
            'train_plots': {'train':{}},
            'train_epochs': [],
            'eval_plots': {'train':{}, 'val':{}, 'test':{}},
            'eval_epochs': []
        }

        eval_data = [
            ('train', train_loader),
            ('val', val_loader),
            ('test', test_loader)
        ]
                        
        eval_epochs = []
        eval_model_weights = {}
        
        for e in range(args.epochs):                       

            train_loader.mode = 'train'
            
            train_utils.run_train_epoch(
                node_args,
                cur_res,                
                net,
                opt,
                train_loader,
                None,
                utils.SEARCH_TRAIN_LOG_INFO,
                e,
                model_train_batch,
            )

            train_loader.mode = 'eval'
            val_loader.mode = 'eval'
            test_loader.mode = 'eval'

            best_ep = eval_utils.run_eval_epoch(
                node_args,
                cur_res,
                net,
                eval_data,
                utils.SEARCH_EVAL_LOG_INFO,
                e,
                grammar,
                model_eval,
                eval_utils.search_model_eval
            )

            if best_ep >= 0:
                best_ep_ind = cur_res['eval_epochs'].index(best_ep)
                res[n] = {
                    k: cur_res['eval_plots']['test'][k][best_ep_ind] \
                    for k in cur_res['eval_plots']['test']
                }

                res[n]['Epoch'] = best_ep
                
                break

            if (e+1) % args.eval_per == 0:
                eval_model_weights[e] = deepcopy(net.state_dict())
        
        utils.log_print("Saving Best Model", args)    

        torch.save(
            eval_model_weights[best_ep],
            f"{args.outpath}/{args.exp_name}/models/{MODEL_TYPE}_net_{n}.pt"
        )

    for n in node_list:
        train_node(n)            
            
    for node, values in res.items():
        utils.log_print(f"Node {node} ({grammar.fi2l[node]}):", args)
        for k,v in values.items():
            utils.log_print(f"    {k} : {v} ", args)
    
        
if __name__ == '__main__':
    arg_list = utils.SEARCH_DEF_ARGS
    args = utils.getArgs(arg_list)
    if args.search_data_path is None:
        args.search_data_path = f'{args.outpath}/{args.exp_name}/prop/search_data'
        
    main(args)
