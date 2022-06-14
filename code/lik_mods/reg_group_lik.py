import os
import torch, utils
from utils import device
from tqdm import tqdm
from copy import deepcopy
import data_utils, train_utils
from grammar import Grammar
import numpy as np
from models import GatedGCN, PointNetPPEnc
import random
import scipy.stats
from focal_loss import FocalLoss
import pc_enc.pc_ae as pc_ae
import pc_enc.pairpc_ae as pairpc_ae
import dgl

MODEL_TYPE = 'reg'

MIN_LL = -100.
LOG_EPS = 1e-40
BS = 16
MAX_SEGMENTS = 128

ADD_SELF_EDGES = True

celoss = FocalLoss(alpha=0.5, gamma = 2., reduction='mean')        

REG_TRAIN_LOG_INFO = [
    ('Loss', 'loss', 'nc'),
    ('Lab Loss', 'lab_loss', 'nc'),
    ('Pur Loss', 'pur_loss', 'nc'),
    ('Lab Accuracy', 'lab_corr', 'lab_total'),
]

def calc_area_sim(area_data, pos_sig, neg_sig):
    match_area = 1e-8
    pos_area = 1e-8
    neg_area = 1e-8

    info = {}

    for seg, key in enumerate(pos_sig):
        info[seg] = key
        pos_area += area_data[seg]

    for seg, key in enumerate(neg_sig):
        neg_area += area_data[seg]
        
        if seg in info and info[seg] == key:
            match_area += area_data[seg]

    ref_area = max(pos_area, neg_area)

    return match_area / ref_area

def getSig(labels):
    z = deepcopy(labels)            
    z.sort()
    return tuple(z)

def regParseDataWithLevel(
    search_data,
    level_info,
    grammar,
    args
):
    reg_data = {
        'shape_inds': [],
        'reg_inds': [],
        'reg_label': [],
        'reg_purity': []
    }

    label_set, lev_l2i, lev_i2l, level_map = level_info

    l = list(zip(search_data.data['inds'], search_data.data['labels'], search_data.data['areas']))

    bucket_data = {
        0.5: {k:[] for k in reg_data.keys()},
        0.6: {k:[] for k in reg_data.keys()},
        0.7: {k:[] for k in reg_data.keys()},
        0.8: {k:[] for k in reg_data.keys()},
        0.9: {k:[] for k in reg_data.keys()},
        1.0: {k:[] for k in reg_data.keys()},
    }
    
    for i, (ind, labels, area_data) in tqdm(enumerate(l), total=len(l)):
                    
        Lsearch_data = np.load(
            f'{args.search_data_path}/{ind}/prop_samples.npz'
        )
        prop_samples = Lsearch_data['prop_samples']

        seen_gt_regs = set()
        seen_regs = set()        

        for ps in [labels]:
            cls = level_map[ps]
            for l in np.unique(cls):
                r = getSig((cls == l).nonzero()[0].tolist())
                seen_gt_regs.add(r)
                seen_regs.add(r)

        _prop_samples = []
        
        for ps in prop_samples:
            P = np.array([grammar.fl2i[grammar.i2l[l]] for l in ps])
            _prop_samples.append(P)
        
        for ps in  _prop_samples:            
            cls = level_map[ps]
            for l in np.unique(cls):
                r = getSig((cls == l).nonzero()[0].tolist())
                seen_regs.add(r)

        flags = {
            0.5: False,
            0.6: False,
            0.7: False,
            0.8: False,
            0.9: False
        }
        
        for reg in list(seen_gt_regs) + list(seen_regs):

            if flags[0.5] and flags[0.6] and flags[0.7] and flags[0.8] and flags[0.9]:
                break
            
            samp_labels = level_map[search_data.labels[i]]
            segments = search_data.segments[i]

            reg_inds = np.concatenate([
                (segments == r).nonzero()[0].reshape(-1) for r in reg
            ], axis = 0)

            if reg_inds.shape[0] == 0:
                continue
            
            else:            
                _reg_label, reg_label_count = scipy.stats.mode(samp_labels[reg_inds])
                reg_label = _reg_label[0]
            
            tars = np.zeros(segments.shape[0])
            label_inds = (samp_labels == reg_label).nonzero()[0]

            tars[label_inds] = 1.0
            tars[reg_inds] = 1.0
            
            purity = reg_label_count[0] * 1.0 / tars.sum().item()

            if purity < 0.5:
                continue

            bucket_pur = int(purity * 10) / 10.

            bdata = bucket_data[bucket_pur]

            # Don't surpass number of regions that max GT can have
            if (len(bdata['shape_inds']) / (i+1.)) > len(level_info[0]):
                flags[bucket_pur] = True
                continue
            
            bdata['shape_inds'].append(i)
            bdata['reg_inds'].append(torch.tensor(reg))
            bdata['reg_label'].append(reg_label)
            bdata['reg_purity'].append(purity)
            
    final_data = {k:[] for k in reg_data.keys()}

    num_per_bucket = len(bucket_data[1.0]['shape_inds'])
            
    for bdata in bucket_data.values():
        rinds = random.sample(list(range(len(bdata['shape_inds']))), min(num_per_bucket, len(bdata['shape_inds'])))
        for k,v in bdata.items():
            for ri in rinds:
                final_data[k].append(v[ri])

    return final_data

class RegNet(torch.nn.Module):
    def __init__(self, args, level_info):
        super(RegNet, self).__init__()

        num_labs = len(level_info[0])
        
        self.lab_net = GatedGCN(
            num_labs,
            args.hidden_dim,
            args.node_dim + args.sem_dim,
            args.edge_dim,
            args.dropout,
            batch_norm = False,
            device = device,
            per_node_pred = False,        
            sem_num=2,
            sem_dim=args.sem_dim,        
        )

        self.pur_net = GatedGCN(
            1,
            args.hidden_dim,
            args.node_dim + args.sem_dim,
            args.edge_dim,
            args.dropout,
            batch_norm = False,
            device = device,
            per_node_pred = False,        
            sem_num=num_labs + 1,
            sem_dim=args.sem_dim,        
        )
        
def getNet(args, level_info):
    return RegNet(args, level_info)

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
        faces, verts.unsqueeze(0), utils.NUM_EVAL_POINTS
    )
        
    samp_segments = segments[face_inds]    
    samp_labels = labels[face_inds]
        
    return samps, samp_segments, samp_labels


class Dataset:
    def __init__(
        self, data, pc_enc, pairpc_enc, grammar, batch_size
    ):

        self.data = data
        self.mode = 'train'                        
        self.batch_size = batch_size
                                
        self.grammar = grammar

        self.graphs = []
        self.seg_labels = []

        self.segments = []
        self.labels = []        
        
        with torch.no_grad():
            for meshes, seg_labels in tqdm(list(zip(self.data['meshes'], self.data['labels']))):
                seg_labels = torch.from_numpy(seg_labels).long()
                
                _, segments, samp_labels = sampleShape(meshes, seg_labels)
                
                self.segments.append(
                    segments[:utils.NUM_EVAL_POINTS].cpu().numpy().astype('int16')
                )

                self.labels.append(
                    samp_labels[:utils.NUM_EVAL_POINTS].cpu().numpy().astype('int16')
                )
                
                part_samps = []
                
                for mesh in meshes:
                    part_samps.append(pc_ae.samplePart(mesh))
                    
                part_samps = torch.stack(part_samps, dim = 0)
                
                center = (part_samps.max(dim=1).values + part_samps.min(dim=1).values) / 2
                part_samps -= center.unsqueeze(1)
                norm = torch.norm(part_samps, dim=2).max(dim=1).values                
                part_samps /= (norm.view(-1,1,1) + 1e-8)

                part_enc = []
                
                for k in range(0, part_samps.shape[0], BS):
                    _part_enc = pc_enc(part_samps[k:k+BS].cuda())
                    part_enc.append(_part_enc.cpu())

                part_enc = torch.cat(part_enc, dim = 0)
                    
                node_feats = torch.cat((part_enc.cpu(), center, norm.view(-1, 1)), dim=1)
                    
                g = dgl.DGLGraph()
                
                src = []
                dst = []                

                if seg_labels.shape[0] == 1 or seg_labels.shape[0] > MAX_SEGMENTS:
                    g.add_nodes(seg_labels.shape[0])
                    if ADD_SELF_EDGES:
                        src += list(range(seg_labels.shape[0]))
                        dst += list(range(seg_labels.shape[0]))
                        edge_feats = part_enc.cpu()
                    
                    g.add_edges(src, dst)

                    g.ndata['node_feats'] = node_feats.cpu()
                    g.edata['edge_feats'] = edge_feats.cpu()
                
                    self.graphs.append(g)                                                
                    self.seg_labels.append(seg_labels.cpu())
                    
                    continue
                
                for i in range(seg_labels.shape[0]):
                    for j in range(seg_labels.shape[0]):
                        if i == j:
                            continue
                        src.append(i)
                        dst.append(j)

                c_shapes = []

                for i,j in zip(src, dst):
                    c_shapes.append(
                        torch.cat((
                            part_samps[i], part_samps[j]
                        ), dim = 0)
                    )

                
                    
                c_shapes = torch.stack(c_shapes, dim =0)

                c = (c_shapes.max(dim=1).values + c_shapes.min(dim=1).values) / 2
                c_shapes -= c.unsqueeze(1)
                norm = torch.norm(c_shapes, dim=2).max(dim=1).values                
                c_shapes /= (norm.view(-1,1,1) + 1e-8)

                oh = torch.zeros(c_shapes.shape[0], c_shapes.shape[1], 2, device= c_shapes.device)
                oh[:,:(c_shapes.shape[1]//2),0] = 1.0
                oh[:,(c_shapes.shape[1]//2):,1] = 1.0

                pair_shapes = torch.cat((c_shapes, oh), dim = 2)
                edge_feats = []
                                
                for k in range(0, pair_shapes.shape[0], BS):
                    _edge_feats = pairpc_enc(pair_shapes[k:k+BS].cuda())
                    edge_feats.append(_edge_feats.cpu())

                edge_feats = torch.cat(edge_feats, dim = 0)
                                
                g.add_nodes(seg_labels.shape[0])

                if ADD_SELF_EDGES:
                    src += list(range(seg_labels.shape[0]))
                    dst += list(range(seg_labels.shape[0]))
                    edge_feats = torch.cat((edge_feats, part_enc.cpu()), dim=0)
                    
                g.add_edges(src, dst)

                g.ndata['node_feats'] = node_feats.cpu()
                g.edata['edge_feats'] = edge_feats.cpu()
                
                self.graphs.append(g)                                                
                self.seg_labels.append(seg_labels.cpu())
        
                                

    def add_level_data(self, node_data, level_info, args):

        self.level_info = level_info
        self.args = args
        
        self.shape_inds = node_data['shape_inds']
        self.reg_inds = node_data['reg_inds']
        self.reg_labels = node_data['reg_label']
        self.reg_puritys = node_data['reg_purity']


    def __iter__(self):

        if self.mode == 'train':                    
            yield from self.train_iter()            
                
        else:
            assert False, f'bad mode {self.mode}'
            

    def train_iter(self):

        inds = list(range(len(self.shape_inds))) 
        
        random.shuffle(inds)
        
        for start in range(0, len(self.shape_inds), self.batch_size):
            end = start + self.batch_size

            binds = inds[start:end]
            
            bshape_inds = [self.shape_inds[bi] for bi in binds]
            
            breg_inds = [self.reg_inds[bi] for bi in binds]
            breg_labels = torch.tensor([self.reg_labels[bi] for bi in binds]).long().to(device)
            breg_puritys = torch.tensor([self.reg_puritys[bi] for bi in binds]).float().to(device)

            bgraphs = []
                                    
            with torch.no_grad():

                for (graph, ireg_inds, ireg_label) in zip(
                    [self.graphs[si] for si in bshape_inds],
                    breg_inds,
                    breg_labels    
                ):                                        
                    _graph = make_graph(
                        graph,
                        ireg_inds,
                        ireg_label,
                    )
                    
                    bgraphs.append(_graph)
                                        
                bgraphs = dgl.batch(bgraphs)
                bgraphs = bgraphs.to(device)

            yield bgraphs, breg_labels, breg_puritys


def make_graph(
    Rgraph, inds, label
):

    graph = deepcopy(Rgraph)
    
    graph.ndata['l_cl'] = torch.zeros(graph.number_of_nodes()).long()
    graph.ndata['p_cl'] = torch.zeros(graph.number_of_nodes()).long()

    graph.ndata['l_cl'][inds] = 1
    graph.ndata['p_cl'][inds] = label.item() + 1

    return graph

def get_probs(net, num_segments, samps, segments, info, num_oh, batch_size):
    shapes = make_shapes(
        num_segments, samps, segments, info, 0., 0., num_oh
    )

    preds = []
    start = 0

    while(start < shapes.shape[0]):
        end = start+batch_size
        preds.append(torch.sigmoid(net(shapes[start:end]).flatten()).cpu())
        start = end
        
    preds = torch.cat(preds, dim=0)

    return preds
    
            
def model_train_batch(batch, net, opt):
    br = {}

    batch_graph, reg_labels, reg_puritys = batch

    node_feats = batch_graph.ndata['node_feats']
    edge_feats = batch_graph.edata['edge_feats']

    batch_graph.ndata['cl'] = batch_graph.ndata['l_cl']
    lab_preds = net.lab_net(batch_graph, node_feats, edge_feats)

    batch_graph.ndata['cl'] = batch_graph.ndata['p_cl']
    pur_preds = torch.sigmoid(net.pur_net(batch_graph, node_feats, edge_feats).flatten())

    lab_loss = celoss(lab_preds, reg_labels)
    pur_loss = (pur_preds - reg_puritys).abs().mean()

    loss = lab_loss + pur_loss
                    
    with torch.no_grad():
                
        lab_corr = (lab_preds.argmax(dim=1) == reg_labels).sum().item()
        lab_total = reg_labels.shape[0] * 1.0

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    br['loss'] = loss.item()
    br['lab_loss'] = lab_loss.item()
    br['pur_loss'] = pur_loss.item()
    br['lab_corr'] = lab_corr
    br['lab_total'] = lab_total
    br['nc'] = 1.

    return br
            
def train_main(args):

    os.system(f'mkdir {args.outpath} > /dev/null 2>&1')    
    os.system(f'mkdir {args.outpath}/{args.exp_name} > /dev/null 2>&1')
    
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
    
    print("Loading Encoders")
    
    pc_enc = PointNetPPEnc(64, input_channels=0, USE_BN=False)
    pc_enc.load_state_dict(torch.load(args.pc_ae_path))
    pc_enc.eval()
    pc_enc.to(device)

    pairpc_enc = PointNetPPEnc(64, input_channels=2, USE_BN=False)
    pairpc_enc.load_state_dict(torch.load(args.pairpc_ae_path))
    pairpc_enc.eval()
    pairpc_enc.to(device)
                    
    print("Making Datasets")
    
    train_loader = Dataset(            
        train_data,
        pc_enc,
        pairpc_enc,
        grammar,
        batch_size = args.batch_size,                
    )
        
    val_loader = Dataset(
        val_data,
        pc_enc,
        pairpc_enc,
        grammar,
        batch_size = args.batch_size,
    )
        
    test_loader = Dataset(            
        test_data,
        pc_enc,
        pairpc_enc,
        grammar,
        batch_size = args.batch_size,
    )

    del pc_enc
    del pairpc_enc
    
    utils.init_model_run(args, MODEL_TYPE)

    res = {
        'train_plots': {'train':{}, 'val': {}, 'test':{}},
        'train_epochs': [],
    }

    lab_model_weights = {}
    pur_model_weights = {}
    
    
    def train_reg(level_info):
        
        print("Parsing Saved Proposals")

        level_train_data = regParseDataWithLevel(
            train_loader, level_info, grammar, args
        )
            
        level_val_data = regParseDataWithLevel(
            val_loader, level_info, grammar, args
        )
        level_test_data = regParseDataWithLevel(
            test_loader, level_info, grammar, args
        )
        
        print("Constructing Node Data")
        
        train_loader.add_level_data(
            level_train_data,
            level_info,
            args,
        )
        val_loader.add_level_data(
            level_val_data,
            level_info,
            args,
        )
        test_loader.add_level_data(
            level_test_data,
            level_info,
            args,
        )        
                    
        net = getNet(args, level_info)
                
        net.to(device)        
        
        opt = torch.optim.Adam(
            net.parameters(),
            lr = args.lr,
            eps = 1e-6
        )        

        best_lab_ep = 0
        best_lab_acc = -1e8
        best_pur_ep = 0
        best_pur_loss = 1e8
                
        for e in range(args.epochs):                       

            train_loader.mode = 'train'
            
            train_utils.run_train_epoch(
                args,
                res,                
                net,
                opt,
                train_loader,
                val_loader,
                REG_TRAIN_LOG_INFO,
                e,
                model_train_batch,
                test_loader
            )

            if (e+1) % args.print_per != 0:
                continue
            
            lab_acc = res['train_plots']['val']['Lab Accuracy'][-1]
            pur_loss = res['train_plots']['val']['Pur Loss'][-1]
            
            if lab_acc > best_lab_acc + args.es_threshold:
                best_lab_acc = lab_acc
                best_lab_ep = e

            if  pur_loss < best_pur_loss - args.es_threshold:
                best_pur_loss = pur_loss
                best_pur_ep = e
            
            if (e - best_pur_ep) > args.es_patience and (e - best_lab_ep) > args.es_patience:
                break
            
            lab_model_weights[e] = deepcopy(net.lab_net.state_dict())
            pur_model_weights[e] = deepcopy(net.pur_net.state_dict())
        
        utils.log_print("Saving Best Model", args)    

        torch.save(
            lab_model_weights[best_lab_ep],
            f"{args.outpath}/{args.exp_name}/models/{MODEL_TYPE}_net_lab.pt"
        )

        torch.save(
            pur_model_weights[best_pur_ep],
            f"{args.outpath}/{args.exp_name}/models/{MODEL_TYPE}_net_pur.pt"
        )

    level_info = getLevel(grammar)

    print(f"LEVEL INFO {level_info}")
    
    train_reg(level_info)
    
    
def getLevel(grammar):
    return grammar.cut_levels[-1]
            
class RegLik:
    def __init__(self, args, grammar):
        args.pc_ae_path = 'pc_enc/chair_pc_ae/models/model.pt'
        args.pairpc_ae_path = 'pc_enc/chair_pairpc_ae/models/model.pt'
        args.hidden_dim = 64
        args.node_dim = 68
        args.edge_dim = 64
        args.sem_dim = 16
        args.lr = 0.00025
        self.args = args
        self.grammar = grammar
        
    def train(self):
        train_main(deepcopy(self.args))

    def load_for_map(self, args, test_data):
        self.cache = {}
        grammar = self.grammar
        args = self.args
    
        level_models = {}

        level_info = getLevel(grammar)

        prev_name = f'{args.load_exp_name}_{args.category}_{args.train_size}_reg'
        net = getNet(args, level_info)
        net.level_info = level_info
        i2lev = np.array([level_info[-1][grammar.fl2i[grammar.i2l[i]]] for \
                          i in range(len(grammar.i2l))])
        net.i2lev = i2lev
        net.lab_net.load_state_dict(torch.load(
            f'{args.outpath}/{prev_name}/{MODEL_TYPE}/models/{MODEL_TYPE}_net_lab.pt'
        ))
        net.pur_net.load_state_dict(torch.load(
            f'{args.outpath}/{prev_name}/{MODEL_TYPE}/models/{MODEL_TYPE}_net_pur.pt'
        ))
        net.eval()

        self.net = net
        
        pc_enc = PointNetPPEnc(64, input_channels=0, USE_BN=False)
        pc_enc.load_state_dict(torch.load(args.pc_ae_path))
        pc_enc.eval()
        pc_enc.to(device)

        pairpc_enc = PointNetPPEnc(64, input_channels=2, USE_BN=False)
        pairpc_enc.load_state_dict(torch.load(args.pairpc_ae_path))
        pairpc_enc.eval()
        pairpc_enc.to(device)

        self.loader = Dataset(            
            test_data,
            pc_enc,
            pairpc_enc,
            grammar,
            batch_size = args.batch_size,
        )

        del pc_enc
        del pairpc_enc


    def calc_lik(self, beams, ind, do_split=False):
        if ind not in self.cache:
            self.cache[ind] = {}
            
        grammar = self.grammar
        args = self.args
        
        graph = self.loader.graphs[ind]
        net = self.net

        net.to(device)
        
        tll = torch.zeros(beams.shape[0]).float()                

        # list of region, label pairs to predict for
        L_ull = []
        L_pinds = []
        L_pmaps = []
        L_pcount = []
        
        pinfo = []
                
        offset = 0
        
        rinfo = net.i2lev[beams]
        
        for l in np.unique(rinfo):
            lregs, l_map = np.unique(rinfo == l, axis=0, return_inverse=True)
            
            ull = torch.zeros(lregs.shape[0]).float()            
            pinds = []
            lcount = []
            for i, _reg in enumerate(lregs):
                if _reg.sum() == 0.:
                    lcount.append(0.)
                    continue
                else:
                    lcount.append(1.)
                    
                reg = _reg.nonzero()[0]                
                if (tuple(reg), l) in self.cache[ind]:
                    ull[i] = self.cache[ind][(tuple(reg), l)]
                else:
                    pinfo.append((reg, l))
                    pinds.append(i + offset)
                                
            L_ull.append(ull)
            L_pmaps.append(l_map+offset)
            L_pinds.append(pinds)
            L_pcount.append(np.array(lcount))
            offset += lregs.shape[0]

        ull = np.concatenate(L_ull, axis=0)
        pcount = np.concatenate(L_pcount, axis=0)
        pinds = np.concatenate(L_pinds, axis=0).astype('long')
        pmaps = np.stack(L_pmaps, axis=0)
                    
        lab_preds = []
        pur_preds = []
                                    
        num_oh = len(net.level_info[0])

        start = 0
        while start < len(pinfo):

            end = start + args.batch_size

            bgraphs = []
                                        
            for bi in range(start, min(end, len(pinfo))):
                ireg_inds, ireg_label = pinfo[bi]
                
                _graph = make_graph(
                    graph,
                    ireg_inds,
                    ireg_label,
                )                        
                bgraphs.append(_graph)
                
            batch_graph = dgl.batch(bgraphs)
            batch_graph = batch_graph.to(device)

            node_feats = batch_graph.ndata['node_feats']
            edge_feats = batch_graph.edata['edge_feats']

            batch_graph.ndata['cl'] = batch_graph.ndata['l_cl']
                        
            lab_pred = net.lab_net(batch_graph, node_feats, edge_feats)

            batch_graph.ndata['cl'] = batch_graph.ndata['p_cl']
            pur_pred = net.pur_net(batch_graph, node_feats, edge_feats)
                    
            blab_preds = torch.clamp(torch.log(lab_pred.softmax(dim=1).cpu() + LOG_EPS), MIN_LL, 0.)
            bpur_preds = torch.clamp(torch.log(torch.sigmoid(pur_pred).flatten().cpu() + LOG_EPS), MIN_LL, 0.)
            
            for ii, bi in enumerate(range(start, min(end, len(pinfo)))):
                ireg_inds, ireg_label = pinfo[bi]
                lab_preds.append(blab_preds[ii, ireg_label])                
                self.cache[ind][(tuple(ireg_inds), ireg_label)] = ((blab_preds[ii, ireg_label] + bpur_preds[ii]) / 2.).item()
                
            pur_preds.append(bpur_preds)                    
            start = end

        if len(lab_preds) > 0 :
            lab_preds = torch.stack(lab_preds, dim=0)
            pur_preds = torch.cat(pur_preds, dim=0)
            ull[pinds] = (lab_preds + pur_preds) / 2.
        else:
            assert pinds.shape[0] == 0
            
        tll = torch.from_numpy(ull[pmaps].sum(axis=0))
        tcount = torch.from_numpy(pcount[pmaps].sum(axis=0))
        
        pred = tll / tcount
        
        p_info = None
        net.to(torch.device('cpu'))

        if do_split:
            return {'reg_all':pred}
        
        return pred, p_info        
