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

# Can add more lambdas here, but this one empirically did the best over a grid-search of values
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
    return preds, seg_pred
    
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

            samp_points, samp_segments, samp_labels = sampleShape(
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
            ]:
                if V.shape[0] < 10000:
                    V = V.repeat( (10000 // V.shape[0]) + 1, 0)[:10000]
                L.append(V)                            

            lhss_pts = torch.from_numpy(self.pts[-1]).to(device)

            dst = ((samp_points.to(device).view(1, -1, 3) - lhss_pts.view(-1, 1, 3)).abs() + 1e-8).norm(dim=2)
            lhss_seg = samp_segments[dst.argmin(dim=1).cpu()].numpy()
            self.plb.append(lhss_seg.reshape(-1,1))        
            
        self.curv = np.stack(self.curv, axis=0)
        self.locpca = np.stack(self.locpca, axis=0)
        self.locvar = np.stack(self.locvar, axis=0)
        self.spinfea = np.stack(self.spinfea, axis=0)
        self.sc = np.stack(self.sc, axis=0)
        self.dhist = np.stack(self.dhist, axis=0)
        self.pts = np.stack(self.pts, axis=0)
        self.ptn = np.stack(self.ptn, axis=0)
        self.plb = np.stack(self.plb, axis=0)

            
    def __iter__(self):

        if self.mode == 'train':
            yield from self.train_iter()
            
        elif self.mode == 'eval':
            yield from self.eval_iter()

        else:
            assert False, f'bad mode {self.mode}'
     
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

                b_plb = torch.from_numpy(self.plb[ind]).long()

                eval_segs = torch.from_numpy(self.eval_segs[ind]).long() 
                eval_labels = torch.from_numpy(self.eval_labels[ind]).long()

                meshes = self.meshes[ind]
            
                
                yield b_curv, b_locpca, b_locvar, b_spinfea, b_sc, b_dhist, b_pts, b_ptn, b_plb, eval_segs, eval_labels, meshes
                
                           


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

    lmbda = LAMBDA
    data = []
    eval_las = []
    for unary_pots, meshes, eval_segs, eval_labels in info:
        preds, seg_pred = make_lhss_preds(unary_pots, meshes, eval_segs, grammar, lmbda)
        eval_las.append(seg_pred.long())
        data.append((preds, eval_labels))

    iou = calc_mIoU([d[1] for d in data], [d[0] for d in data], grammar)
    metric_result['all_iou'] = iou        
    metric_result['nc'] = 1
    
    return metric_result, eval_las

def main(args):
        
    utils.init_model_run(args, MODEL_TYPE)
    
    grammar = Grammar(args.category)

    test_data = data_utils.load_dataset(
        f'{args.data_path}/{args.category}',
        f'{args.category}/split.json',
        f'test_{args.eval_size}',
    )
            
    test_loader = Dataset(
        test_data,
        grammar,
        args.batch_size,
    )

    test_loader.mode = 'eval'

    net = LHSSNet(len(grammar.terminals))
    
    net.load_state_dict(torch.load(
        f'model_output/lhss_{args.category}_{args.train_size}/lhss/models/lhss_net.pt'
    ))
    net.eval()
    net.to(device)
    
    eval_res, _ = lhss_model_eval(
        args,
        test_loader,
        net,
        0,
        grammar,
        'test',
        batch_model_eval,        
    )

    utils.log_print(f"~~ TEST EVAL RES ~~", args)
    
    utils.print_results(
        LHSS_PROP_LOG_INFO,
        eval_res,
        args
    )
        
if __name__ == '__main__':

    arg_list = []    
    args = utils.getArgs(arg_list)
    with torch.no_grad():
        main(args)
    


