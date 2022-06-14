import os
import eval_sem_label_models as esm
import train_sem_label_models as ts
import torch, utils
from utils import device
from tqdm import tqdm
from copy import deepcopy
import numpy as np

VERBOSE = False

def make_shapes(num_segments, samps, segments, seg_map, _all_info, grammar, name):

    all_info = []

    for _, in_parts, c_labels in _all_info:

        if 'geom' in name:
            c_labels = [None] * len(in_parts)

        all_info.append((in_parts, c_labels))

    return ts.make_shapes_base(
        num_segments, samps, segments, seg_map, all_info, grammar, name, 0., 0.
    )

def getFastSig(l, inf, mt):
    
    if mt == 'geom':
        ip = inf.nonzero()[0]        
        cls = None
        
    elif mt == 'lay':
        ip = (inf >= 0.).nonzero()[0]
        cls = inf[ip]

    if ip.shape[0] == 0:
        return None
        
    return (l, tuple(ip), tuple(cls) if cls is not None else None)    

def fast_get_search_probs(
    samps, segments, seg_map, beams,
    node_models, grammar, args, model_type, ind, DEF
):
    
    res = torch.zeros(beams.shape[0]).float()
    num_preds = torch.zeros(beams.shape[0]).float() + 1e-8

    L_pinfo = {}
    L_pinds = {}
    L_uinds = {}
    L_ull = {}
    
    for l in node_models.keys():
                
        n2c = grammar.n2cl[l]
        rinfo = n2c[DEF.Mi2f[beams.astype('long')]]

        if model_type == 'geom':
            rinfo = (rinfo >= 0)
            
        uinfo, uinds = np.unique(rinfo,axis=0,return_inverse=True)

        ull = torch.zeros(uinfo.shape[0]).float()
        pinfo = []
        pinds = []
        pcounts = []
        
        for i, uinf in enumerate(uinfo):
            sig = getFastSig(l, uinf, model_type)

            if sig is None:
                pcounts.append(0.)                
                continue
            else:
                pcounts.append(1.)
            
            if sig in DEF.cache[ind]:
                ull[i] = DEF.cache[ind][sig]
            else:
                pinfo.append(sig)
                pinds.append(i)

        pcounts = torch.tensor(pcounts)
        num_preds += pcounts[uinds]
        
        L_pinfo[l] = pinfo
        L_pinds[l] = pinds
        L_uinds[l] = uinds
        L_ull[l] = ull

    for l, net in node_models.items():
        pinfo = L_pinfo[l]
        pinds = L_pinds[l]
        uinds = L_uinds[l]
        ull = L_ull[l]
        
        start = 0
        preds = []

        while(start < len(pinfo)):
            end = start+args.batch_size


            bshapes = make_shapes(
                beams.shape[1],
                samps,
                segments,
                seg_map,
                pinfo[start:end],
                grammar,
                model_type
            )
            
            raw_preds = net(bshapes).flatten().cpu()
            
            ll_preds = torch.clamp(torch.log(
                torch.sigmoid(raw_preds) + esm.LOG_EPS
            ), esm.MIN_LL, 0.).flatten()

            preds.append(ll_preds)

            for sig, p in zip(pinfo[start:end], ll_preds):
                DEF.cache[ind][sig] = p.item()
            
            start = end

        if len(preds) > 0:
            preds = torch.cat(preds, dim =0)
            ull[pinds] = preds        
        
        res += ull[uinds]
        
    return (res / num_preds)

class SemLik:
    def __init__(self, args, grammar, alpha = 1.0, beta = 1.0):
        self.alpha = 1.0
        self.beta = 1.0
        self.args = args
        self.grammar = grammar

    def train(self):
        args = deepcopy(self.args)        
        ts.main(args)

    def load_for_map(self, args, test_data):
        self.cache = {}
        
        grammar = self.grammar
        args = self.args

        self.Mi2f = np.array([grammar.fl2i[grammar.i2l[i]] for i in range(len(grammar.i2l.keys()))])
        
        geom_node_list = grammar.get_geom_node_list()
        lay_node_list = grammar.get_lay_node_list()
    
        node_geom_models = {}
        node_lay_models = {}    

        prev_name = f'{args.load_exp_name}_{args.category}_{args.train_size}_sem'
        
        for n in geom_node_list:
            try:
                geom_net = ts.getNodeNet(args, grammar, 'geom')
                geom_net.load_state_dict(torch.load(
                    f'{args.outpath}/{prev_name}/geom/models/geom_net_{n}.pt'
                ))
                geom_net.to(device)
                geom_net.eval()
                node_geom_models[n] = geom_net
                if VERBOSE:
                    utils.log_print(f"Loaded geom net for {n}", args)
            
            except Exception as e:
                utils.log_print(f"Failed to load geom net for {n} with {e}", args)

        for n in lay_node_list:
            try:
                lay_net = ts.getNodeNet(args, grammar, 'lay')            
                lay_net.load_state_dict(torch.load(
                    f'{args.outpath}/{prev_name}/lay/models/lay_net_{n}.pt'
                ))
                lay_net.to(device)
                lay_net.eval()
                node_lay_models[n] = lay_net
                if VERBOSE:
                    utils.log_print(f"Loaded lay net for {n}", args)

            except Exception as e:
                utils.log_print(f"Failed to load lay net for {n} with {e}", args)

        self.node_geom_models = node_geom_models
        self.node_lay_models = node_lay_models
                
        self.loader = esm.Dataset(
            test_data,
            grammar,
            args
        )

    def calc_lik(self, beams, ind, do_split=False):

        if ind not in self.cache:
            self.cache[ind] = {}
        
        geom_ll = fast_get_search_probs(
            torch.from_numpy(self.loader.samps[ind]).float().to(device),
            torch.from_numpy(self.loader.segments[ind]).long().to(device),
            self.loader.seg_maps[ind],
            beams,
            self.node_geom_models,
            self.grammar,
            self.args,
            'geom',
            ind,
            self
        )
                
        lay_ll = fast_get_search_probs(
            torch.from_numpy(self.loader.samps[ind]).float().to(device),
            torch.from_numpy(self.loader.segments[ind]).long().to(device),
            self.loader.seg_maps[ind],
            beams,
            self.node_lay_models,
            self.grammar,
            self.args,
            'lay',
            ind,
            self
        )
        
        preds = geom_ll + lay_ll

        p_info = {}

        if do_split:
            return {'sem_geom': geom_ll, 'sem_lay': lay_ll}
        
        return preds, p_info        
        
