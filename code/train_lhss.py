from grammar import Grammar
import sys
sys.path.append('lhss')
sys.path.append('lhss/pygco')
from mrf import eval_mrf
import torch
import data_utils
from tqdm import tqdm
from models import LHSSNet
import train_utils, eval_utils
import utils
from utils import device
import numpy as np
from eval_utils import calc_mIoU
from copy import deepcopy

LAMBDA = 0.1

MODEL_TYPE = 'lhss'

celoss = torch.nn.CrossEntropyLoss()

import lhss.feat_code.calc_feat as cf

LHSS_PROP_LOG_INFO = [    
    ('mIoU', 'all_iou', 'nc'),
]

def make_lhss_preds(unary_pots, meshes, eval_segs, grammar, lmbda):
    seg_pred = torch.from_numpy(eval_mrf(meshes, unary_pots.numpy(), grammar, lmbda))
    preds = seg_pred[eval_segs]
    return preds
    
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
    

class Dataset:
    def __init__(
        self, data, grammar, batch_size, train_pts = None
    ):
        
        self.mode = 'train'

        self.batch_size = batch_size
        self.train_pts = train_pts
        
        self.curv = []
        self.locpca = []
        self.locvar = []
        self.spinfea = []
        self.sc = []
        self.dhist = []
        self.pts = []
        self.ptn = []
        self.plb = []
        
        self.point_labels = []

        self.eval_segs = []
        self.eval_labels = []

        self.meshes = []
        
        for i, (ind, meshes, labels) in tqdm(
            enumerate(zip(data['inds'], data['meshes'], data['labels'])),
            total=len(data['labels'])
        ):            
                        
            res = cf.load_lhss_data(meshes, ind)

            self.meshes.append(meshes)
            
            seg_labels = grammar.level_map[labels]

            _, samp_segments, samp_labels = sampleShape(
                meshes,
                seg_labels
            )
            n_samp_segments = samp_segments.numpy().astype('int16')
            n_samp_labels = samp_labels.numpy().astype('int16')
            
            self.eval_segs.append(n_samp_segments)
            self.eval_labels.append(n_samp_labels)                        

            for L, V in [
                    (self.curv, res['curv']),
                    (self.locpca, res['locpca']),
                    (self.locvar, res['locvar']),
                    (self.spinfea, res['spinfea']),
                    (self.sc, res['sc']),
                    (self.dhist, res['dhist']),
                    (self.pts, res['pts']),
                    (self.ptn, res['ptn']),
                    (self.plb, res['plb']),                        
            ]:
                if V.shape[0] < 10000:
                    V = V.repeat( (10000 // V.shape[0]) + 1, 0)[:10000]
                L.append(V)                            

            segments = self.plb[-1] - 1            
            point_labels = seg_labels[segments]
            
            self.point_labels.append(point_labels)

        
            
        self.curv = np.stack(self.curv, axis=0)
        self.locpca = np.stack(self.locpca, axis=0)
        self.locvar = np.stack(self.locvar, axis=0)
        self.spinfea = np.stack(self.spinfea, axis=0)
        self.sc = np.stack(self.sc, axis=0)
        self.dhist = np.stack(self.dhist, axis=0)
        self.pts = np.stack(self.pts, axis=0)
        self.ptn = np.stack(self.ptn, axis=0)
        self.plb = np.stack(self.plb, axis=0)

        self.point_labels = np.stack(self.point_labels, axis=0)
            
    def __iter__(self):

        if self.mode == 'train':
            yield from self.train_iter()
            
        elif self.mode == 'eval':
            yield from self.eval_iter()

        else:
            assert False, f'bad mode {self.mode}'

    def train_iter(self):

        inds = torch.randperm(self.curv.shape[0]).numpy()

        for i in range(0, self.curv.shape[0], self.batch_size):        
            with torch.no_grad():

                binds = inds[i:i+self.batch_size]

                pt_inds = torch.randperm(10000).numpy()[:self.train_pts]
                
                b_curv = torch.from_numpy(self.curv[binds][:,pt_inds]).float().to(device)
                b_locpca = torch.from_numpy(self.locpca[binds][:,pt_inds]).float().to(device)
                b_locvar = torch.from_numpy(self.locvar[binds][:,pt_inds]).float().to(device)
                b_spinfea = torch.from_numpy(self.spinfea[binds][:,pt_inds]).float().to(device)
                b_sc = torch.from_numpy(self.sc[binds][:,pt_inds]).float().to(device)
                b_dhist = torch.from_numpy(self.dhist[binds][:,pt_inds]).float().to(device)
                b_pts = torch.from_numpy(self.pts[binds][:,pt_inds]).float().to(device)
                b_ptn = torch.from_numpy(self.ptn[binds][:,pt_inds]).float().to(device)
                
                b_labels = torch.from_numpy(self.point_labels[binds][:,pt_inds]).long().to(device)
                
            yield b_curv, b_locpca, b_locvar, b_spinfea, b_sc, b_dhist, b_pts, b_ptn, b_labels
 
    
    def eval_iter(self):

        inds = list(range(self.curv.shape[0]))
        
        while len(inds) > 0:                        
            with torch.no_grad():
                ind = inds.pop(0)
                b_curv = torch.from_numpy(self.curv[ind]).float().to(device)
                b_locpca = torch.from_numpy(self.locpca[ind]).float().to(device)
                b_locvar = torch.from_numpy(self.locvar[ind]).float().to(device)
                b_spinfea = torch.from_numpy(self.spinfea[ind]).float().to(device)
                b_sc = torch.from_numpy(self.sc[ind]).float().to(device)
                b_dhist = torch.from_numpy(self.dhist[ind]).float().to(device)
                b_pts = torch.from_numpy(self.pts[ind]).float().to(device)
                b_ptn = torch.from_numpy(self.ptn[ind]).float().to(device)

                b_plb = torch.from_numpy(self.plb[ind]).long() - 1

                eval_segs = torch.from_numpy(self.eval_segs[ind]).long() 
                eval_labels = torch.from_numpy(self.eval_labels[ind]).long()

                meshes = self.meshes[ind]
            
                
                yield b_curv, b_locpca, b_locvar, b_spinfea, b_sc, b_dhist, b_pts, b_ptn, b_plb, eval_segs, eval_labels, meshes
                
                           
def model_train_batch(batch, net, opt):
    

    b_curv, b_locpca, b_locvar, b_spinfea, b_sc, b_dhist, b_pts, b_ptm, b_labels = batch
    
    br = {}

    _preds = net(b_curv, b_locpca, b_locvar, b_spinfea, b_sc, b_dhist, b_pts, b_ptm)

    preds = _preds.view(-1, _preds.shape[2])
    labels = b_labels.flatten()
        
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

def batch_model_eval(
    net, b_curv, b_locpca, b_locvar, b_spinfea,
    b_sc, b_dhist, b_pts, b_ptn, b_plb, num_segs
):

    point_preds = torch.softmax(net(
        b_curv.unsqueeze(0),
        b_locpca.unsqueeze(0),
        b_locvar.unsqueeze(0),
        b_spinfea.unsqueeze(0),
        b_sc.unsqueeze(0),
        b_dhist.unsqueeze(0),
        b_pts.unsqueeze(0),
        b_ptn.unsqueeze(0)
    )[0,:,:], dim = 1)

    plb = b_plb[:,0]

    unary_pots = torch.zeros(num_segs, point_preds.shape[1])

    for i in range(num_segs):
        p = point_preds[(plb == i).nonzero().flatten()]
        if p.shape[0] > 0:
            unary_pots[i] = p.mean(dim=0)
        
    return unary_pots
    

def lhss_model_eval(
    args,
    loader,
    net,
    e,
    grammar,
    name,
    model_eval_fn,
):
        
    info = []

    corr = 0
    total = 0

    total_ll = 0.
    
    for count, (batch) in enumerate(loader):

        b_curv, b_locpca, b_locvar, b_spinfea, b_sc, b_dhist, b_pts, b_ptn, b_plb, eval_segs, eval_labels, meshes = batch

        unary_pots = batch_model_eval(
            net, b_curv, b_locpca, b_locvar, b_spinfea,
            b_sc, b_dhist, b_pts, b_ptn, b_plb, len(meshes)
        )
                                        
        info.append((unary_pots.cpu(), meshes, eval_segs.cpu(), eval_labels.cpu()))

    metric_result = {}
    
    if name == 'val':
        best_lmbda = LAMBDA
        best_miou = 0
        for lmbda in [LAMBDA]:
            data = []
            for unary_pots, meshes, eval_segs, eval_labels in info:
                preds = make_lhss_preds(unary_pots, meshes, eval_segs, grammar, lmbda)
                data.append((preds, eval_labels))
                       
            iou = calc_mIoU([d[1] for d in data], [d[0] for d in data], grammar)

            if iou > best_miou:
                best_lmbda = lmbda
                best_miou = iou
                
        net.best_lmbda = best_lmbda
        print(best_lmbda)
        metric_result['all_iou'] = best_miou
        
    else:
        lmbda = net.best_lmbda
        data = []
        for unary_pots, meshes, eval_segs, eval_labels in info:
            preds = make_lhss_preds(unary_pots, meshes, eval_segs, grammar, lmbda)
            data.append((preds, eval_labels))

        iou = calc_mIoU([d[1] for d in data], [d[0] for d in data], grammar)
        metric_result['all_iou'] = iou
        
    metric_result['nc'] = 1
    
    return metric_result

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
        args.train_pts
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

    net = LHSSNet(len(grammar.terminals))
            
    net.to(device)
    
    opt = torch.optim.Adam(
        net.parameters(),
        betas=(0.5, 0.999),
        lr = args.lr,
        eps = 1e-6,
        weight_decay=1e-4
    )        
    
    res = {
        'train_plots': {'train':{}},
        'train_epochs': [],
        'eval_plots': {'train':{}, 'val':{}, 'test':{}},
        'eval_epochs': []
    }

    save_model_weights = {}
        
    eval_data = [
        ('val', val_loader),
        ('train', train_loader),        
        ('test', test_loader)
    ]
        
    for e in range(args.epochs):

        train_loader.mode = 'train'
        
        train_utils.run_train_epoch(
            args,
            res,
            net,
            opt,
            train_loader,
            None,
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
            LHSS_PROP_LOG_INFO,
            e,
            grammar,
            batch_model_eval,
            lhss_model_eval
        )
        
        if best_ep >= 0:
            break

        if (e+1) % args.eval_per == 0:
            save_model_weights[e] = deepcopy(net.state_dict())
            
    torch.save(
        save_model_weights[best_ep],
        f"{args.outpath}/{args.exp_name}/models/{MODEL_TYPE}_net.pt"
    )

        
if __name__ == '__main__':
    #test_eval()

    arg_list = [
        ('-b', '--batch_size', 30,  int),
        ('-lr', '--lr', 3e-4,  float),
        ('-prp', '--print_per', 5,  int),
        ('-evp', '--eval_per', 5,  int),
        ('-ep', '--epochs', 1000,  int),
        ('-trnp', '--train_pts', 2000,  int),
        ('-esp', '--es_patience', 40,  int),
        ('-est', '--es_threshold', 0.001,  float),
        ('-esm', '--es_metric', 'mIoU',  str),
    ]
    
    args = utils.getArgs(arg_list)
    main(args)
    


