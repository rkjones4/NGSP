import sys
sys.path.append('../')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import PointNetPPEnc
import data_utils
import utils
from tqdm import tqdm
import json
from utils import device
import numpy as np
import faiss
import time
import matplotlib.pyplot as plt

from multiprocessing import Pool
import multiprocessing
NUM_CORES =  multiprocessing.cpu_count()

DIM = 64
NUM_POINTS = 1024
EVAL_PER = 1
EPOCHS = 1000
BATCH_SIZE = 32
SCALE_WEIGHT = 0.1
# PC encoder is trained on just chair parts
CAT = 'chair'
LR = 0.0001
MAX_SHAPES = 10000

class Dataset:
    def __init__(
        self, data, scale_weight=0.
    ):
        self.scale_weight = scale_weight
                        
        self.samps = []        
                        
        for meshes in tqdm(data):            
            for mesh in meshes:
                with torch.no_grad():                
                    samps = samplePart(mesh)
                    self.samps.append(samps.numpy().astype('float16'))
                    
        self.samps = np.stack(self.samps,axis=0)

    def __iter__(self):

        inds = torch.randperm(self.samps.shape[0]).numpy()

        for i in range(0, self.samps.shape[0], BATCH_SIZE):        
            with torch.no_grad():
                
                binds = inds[i:i+BATCH_SIZE]                
                b_shapes = torch.from_numpy(self.samps[binds]).float().to(device)                
                scale = (torch.randn(b_shapes.shape[0], 1, b_shapes.shape[2], device=device) * self.scale_weight) + 1.                
                a_shapes = (b_shapes * scale)
                c = (a_shapes.max(dim=1).values + a_shapes.min(dim=1).values) / 2
                a_shapes -= c.unsqueeze(1)
                norm = torch.norm(a_shapes, dim=2).max(dim=1).values                
                a_shapes /= (norm.view(-1,1,1) + 1e-8)
                
            yield a_shapes


def samplePart(part):
        
    verts = torch.from_numpy(part[0]).float().to(device)
    faces = torch.from_numpy(part[1]).long().to(device)
                    
    samps, _, _ = utils.sample_surface(
        faces, verts.unsqueeze(0), NUM_POINTS
    )
            
    samps = samps.cpu()
                
    return samps

def robust(var, dim=2):
    return ((var ** 2).sum(dim=dim) + 1e-8).sqrt()

class ChamferLoss(nn.Module):
    def __init__(self, device):
        super(ChamferLoss, self).__init__()
        self.device = device
        self.gpu_id = torch.cuda.current_device()
        self.res = faiss.StandardGpuResources()

    def build_nn_index(self, database):
        index_cpu = faiss.IndexFlatL2(3)
        index = faiss.index_cpu_to_gpu(self.res, self.gpu_id, index_cpu)
        index.add(database)
        return index

    def search_nn(self, index, query):
        D, I = index.search(query, 1)

        D_var = torch.from_numpy(np.ascontiguousarray(D))
        I_var = torch.from_numpy(np.ascontiguousarray(I).astype(np.int64))
        if self.gpu_id >= 0:
            D_var = D_var.to(self.device)
            I_var = I_var.to(self.device)

        return D_var, I_var

    def forward(self, predict_pc, gt_pc):
        predict_pc_size = predict_pc.size()
        gt_pc_size = gt_pc.size()

        predict_pc_np = np.ascontiguousarray(
            torch.transpose(predict_pc.data.clone(), 1, 2).cpu().numpy()
        )  # BxMx3
        gt_pc_np = np.ascontiguousarray(
            torch.transpose(gt_pc.data.clone(), 1, 2).cpu().numpy()
        )  # BxNx3

        # selected_gt: Bxkx3xM
        selected_gt_by_predict = torch.FloatTensor(
            predict_pc_size[0], 1, predict_pc_size[1], predict_pc_size[2]
        )
        # selected_predict: Bxkx3xN
        selected_predict_by_gt = torch.FloatTensor(
            gt_pc_size[0], 1, gt_pc_size[1], gt_pc_size[2]
        )

        if self.gpu_id >= 0:
            selected_gt_by_predict = selected_gt_by_predict.to(self.device)
            selected_predict_by_gt = selected_predict_by_gt.to(self.device)

        # process each batch independently.
        for i in range(predict_pc_np.shape[0]):
            index_predict = self.build_nn_index(predict_pc_np[i])
            index_gt = self.build_nn_index(gt_pc_np[i])

            # database is gt_pc, predict_pc -> gt_pc -----------------------------------------------------------
            _, I_var = self.search_nn(index_gt, predict_pc_np[i])

            selected_gt_by_predict[i, 0, ...] = gt_pc[i].index_select(
                1, I_var[:, 0]
            )

            # database is predict_pc, gt_pc -> predict_pc -------------------------------------------------------
            _, I_var = self.search_nn(index_predict, gt_pc_np[i])

            selected_predict_by_gt[i, 0, ...] = predict_pc[i].index_select(
                1, I_var[:, 0]
            )

        # selected_gt(Bxkx3xM) vs predict_pc(Bx3xM)
        forward_loss_element = robust(
            selected_gt_by_predict
            - predict_pc.unsqueeze(1).expand_as(selected_gt_by_predict)
        )
        forward_loss = forward_loss_element.mean()

        # selected_predict(Bxkx3xN) vs gt_pc(Bx3xN)
        backward_loss_element = robust(
            selected_predict_by_gt
            - gt_pc.unsqueeze(1).expand_as(selected_predict_by_gt)
        )  # BxkxN
        backward_loss = backward_loss_element.mean()
        
        return forward_loss + backward_loss
        
class PartDecoder(nn.Module):

    def __init__(self, feat_len, num_point):
        super(PartDecoder, self).__init__()
        self.num_point = num_point

        self.mlp1 = nn.Linear(feat_len, feat_len * 4)
        self.mlp2 = nn.Linear(feat_len * 4, feat_len * 16)
        self.mlp3 = nn.Linear(feat_len * 16, num_point*3)

    def forward(self, net):
        net = torch.relu(self.mlp1(net))
        net = torch.relu(self.mlp2(net))
        net = self.mlp3(net).view(-1, self.num_point, 3)

        return net

def load_data(
    ind,
):
    
    j = json.load(open(f'../../data/{CAT}/{ind}/data.json'))
    parts = j['parts']

    with Pool(NUM_CORES) as p:
        meshes = p.map(utils.loadAndCleanObj, parts)
    
    return meshes
    
def load_datasets():

    split_info = json.load(open(f'../data_splits/{CAT}/split.json'))

    data_dir = f'../../data/{CAT}'

    train_inds = set(list(os.listdir(data_dir)))

    train_inds = train_inds - set(split_info['val'])
    train_inds = train_inds - set(split_info['test'])

    val_inds = set(split_info['val'])
    val_inds = val_inds - set(split_info['test'])

    train_inds = list(train_inds)
    val_inds = list(val_inds)
    train_inds.sort()
    val_inds.sort()
    
    train_data = [load_data(i) for i in tqdm(train_inds[:MAX_SHAPES])]
    val_data = [load_data(i) for i in tqdm(val_inds[:MAX_SHAPES])]

    return train_data, val_data
    
def train(outdir):

    train_data, val_data = load_datasets()

    train_loader = Dataset(train_data, SCALE_WEIGHT)
    val_loader = Dataset(val_data, 0.)
    
    enc = PointNetPPEnc(DIM, input_channels=0, USE_BN=False)
    enc.to(device)
    dec = PartDecoder(DIM, NUM_POINTS)
    dec.to(device)

    dec_opt = torch.optim.Adam(
        dec.parameters(),
        lr = LR,
    )

    enc_opt = torch.optim.Adam(
        enc.parameters(),
        lr = LR,
    )

    chamfer = ChamferLoss(device)

    res = {
        'epochs': [],
        'train_cd': [],
        'val_cd': []        
    }
    
    
    for ep in range(EPOCHS):
        t = time.time()
        tcl = 0.
        count = 0
        j = 0
        for i, batch in enumerate(train_loader):
            
            code = enc(batch)
            out = dec(code)
            
            loss = chamfer(
                batch.transpose(1,2),
                out.transpose(1,2)
            )

            dec_opt.zero_grad()
            enc_opt.zero_grad()
                    
            loss.backward()
            
            dec_opt.step()
            enc_opt.step()
                
            tcl += loss.detach().item()

            if (ep+1) % EVAL_PER == 0 and count < 1:
                count += 1
                tar = batch.view(-1, NUM_POINTS, 3)
                pred = out.view(-1, NUM_POINTS, 3)
                for s in range(pred.shape[0]):
                    utils.writeSPC(pred[s], f"{outdir}/train/{ep}_ep_{j}_pred.obj")
                    utils.writeSPC(tar[s], f"{outdir}/train/{ep}_ep_{j}_tar.obj")
                    j += 1
            
        tcl /= i+1
        vcl = 0.

        j = 0
        count = 0
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                loss = 0.
                code = enc(batch)
                out = dec(code)

                closs = chamfer(
                    batch.transpose(1,2),
                    out.transpose(1,2)
                )
                
                vcl += closs.detach().item()

                if (ep+1) % EVAL_PER == 0 and count < 1:
                    count += 1
                    tar = batch.view(-1, NUM_POINTS, 3)
                    pred = out.view(-1, NUM_POINTS, 3)
                    for s in range(pred.shape[0]):
                        utils.writeSPC(pred[s], f"{outdir}/val/{ep}_ep_{j}_pred.obj")
                        utils.writeSPC(tar[s], f"{outdir}/val/{ep}_ep_{j}_tar.obj")
                        j += 1

                if (ep+1) % EVAL_PER == 0:
                    torch.save(enc.state_dict(), f"{outdir}/models/enc_state_dict_{ep}.pt")
                    torch.save(dec.state_dict(), f"{outdir}/models/dec_state_dict_{ep}.pt")
                        
        vcl /= i+1        
        print(f"Epoch {ep}, time: {time.time() - t} | TRAIN : Chamfer {tcl} | VAL : Chamfer {vcl}")
        res['epochs'].append(ep)
        res['train_cd'].append(tcl)
        res['val_cd'].append(vcl)

        plt.clf()

        plt.plot(
            res['epochs'],
            res['train_cd'],
            label = 'train'
        )
        plt.plot(
            res['epochs'],
            res['val_cd'],
            label = 'val'
        )
        plt.legend()
        plt.grid()
        plt.savefig(f'{outdir}/plots/CD.png')
        
    
if __name__ == '__main__':
    outdir = sys.argv[1]
    os.system(f'mkdir {outdir}')
    os.system(f'mkdir {outdir}/train')
    os.system(f'mkdir {outdir}/val')
    os.system(f'mkdir {outdir}/models')
    os.system(f'mkdir {outdir}/plots')
    train(outdir)
