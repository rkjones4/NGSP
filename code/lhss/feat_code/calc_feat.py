import os
import torch
import numpy as np
from scipy.io import loadmat
import json
import utils

CACHE_DIR = 'cache_lhss/'
os.system(f'mkdir {CACHE_DIR} > /dev/null 2>&1')
MAX_PTS = 10000

def calcNorms(nv, nf):

    vs = torch.from_numpy(nv).float().unsqueeze(0)
    faces = torch.from_numpy(nf).long()
    
    face_normals = torch.cross(
        vs[:, faces[:, 1], :] - vs[:, faces[:, 0], :],
        vs[:, faces[:, 2], :] - vs[:, faces[:, 1], :],
        dim=2,
    )

    face_areas = torch.norm(face_normals, dim=2) + 1e-8
    face_normals = (face_normals / face_areas[:, :, None])[0]

    return face_normals.numpy()
    

def writeJson(meshes, name):
    av = []
    af = []
    an = []
    ag = []
    
    offset = 1.
    
    for i, (v, f) in enumerate(meshes):
        ns = calcNorms(v, f)

        av.append(v)
        af.append(f + offset)
        an.append(ns)
        ag.append([i+1 for _ in range(f.shape[0])])               
        offset += v.shape[0]

    av = np.concatenate(av, axis=0)
    af = np.concatenate(af, axis=0)
    an = np.concatenate(an, axis=0)
    ag = np.concatenate(ag, axis=0)

    res = {
        'verts': av.tolist(),
        'faces': af.tolist(),
        'normals': an.tolist(),
        'groups': ag.tolist()
    }

    json.dump(res, open(name, 'w'))
        
def loadRes(outname):
    d = loadmat(outname)['feas']

    res = {}
    for key in ['pts', 'ptn', 'plb', 'curv', 'locpca', 'locvar', 'spinfea', 'sc', 'dhist']:
        res[key] = d[key][0,0]
        
    return res
    
    
def gen_data(meshes, ind):
        
    jname = f'tmp/lhss_{ind}.json'
    mname = f'tmp/lhss_{ind}.mat'
    writeJson(meshes, jname)

    with utils.SuppressStream():    
        os.system(f'julia lhss/feat_code/calc_feat.jl {jname} {mname}')
        
    res = loadRes(mname)
    
    os.system(f'rm {jname} > /dev/null 2>&1')
    os.system(f'rm {mname} > /dev/null 2>&1')

    return res
    
def load_lhss_data(meshes, ind, mode = None):
    fls = os.listdir(CACHE_DIR)

    if f'{ind}.npz' in fls:
        res = np.load(f'{CACHE_DIR}/{ind}.npz')
        return res
    else:
        
        res = gen_data(meshes, ind)

        rinds = torch.randperm(res['plb'].shape[0])[:MAX_PTS].numpy().astype('long')
        
        np.savez(
            f'{CACHE_DIR}/{ind}.npz',
            pts = res['pts'][rinds].astype('float16'),
            ptn = res['ptn'][rinds].astype('float16'),
            plb = res['plb'][rinds].astype('int16'),
            curv = res['curv'][rinds].astype('float16'),
            locpca = res['locpca'][rinds].astype('float16'),
            locvar = res['locvar'][rinds].astype('float16'),
            spinfea = res['spinfea'][rinds].astype('float16'),
            sc = res['sc'][rinds].astype('float16'),
            dhist = res['dhist'][rinds].astype('float16'),
        )
        del res
        res = np.load(f'{CACHE_DIR}/{ind}.npz')
        return res
    
        
