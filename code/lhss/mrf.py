import sys
sys.path.append('..')
sys.path.append('lhss')

from lhss.pygco.mrf_solve import mrf_solve
import utils
import numpy as np
import torch
import math

NUM_SAMPS_PER_REGION = 100

def sampleShape(v, f):

    verts = []
    faces = []

    verts = torch.from_numpy(v).float()
    faces = torch.from_numpy(f).long()
            
    samps, _, normals = utils.sample_surface(
        faces, verts.unsqueeze(0), NUM_SAMPS_PER_REGION
    )
        
    samps = samps.cpu().numpy()
    normals = normals.cpu().numpy()
    
    return samps, normals


def getRegMeanInfo(v,f):

    points, normals = sampleShape(v, f)
    
    mp = points.mean(axis=0)
    mn = normals.mean(axis=0)
        
    return mp, mn
    
def makeMRF(meshes, reg_dist):

    reg_info = [getRegMeanInfo(v,f) for v,f in meshes]

    reg_points = np.stack([mp for mp,mn in reg_info])
    reg_norms = np.stack([mn for mp,mn in reg_info])

    reg_norms /= np.linalg.norm(reg_norms, axis=1).reshape(-1, 1)    
    
    points_dist = np.linalg.norm(np.abs(reg_points.reshape(-1, 1, 3) - reg_points.reshape(1, -1, 3)) + 1e-8, axis=2)
    
    norms_dist = reg_norms @ reg_norms.T
    
    geo_dist = points_dist + (.3 *  (1 - np.abs(norms_dist)))

    emb_dist = np.linalg.norm(np.abs(reg_dist.reshape(len(meshes), 1, -1) - reg_dist.reshape(1, len(meshes), -1)) + 1e-8, axis=2)

    geo_sigma = np.sort(geo_dist,axis=1)[:,min(4, len(meshes)-1)].mean()

    emb_sigma = np.sort(emb_dist,axis=1)[:,min(4, len(meshes)-1)].mean()

    geo_sim = np.exp(-1 * geo_dist / geo_sigma / geo_sigma / 2.)
    emb_sim = np.exp(-1 * emb_dist / emb_sigma / emb_sigma / 2.)

    full_sim = geo_sim + emb_sim
        
    patchnn = min(30, math.ceil(0.02 * len(meshes)))
    
    edges = []
    edge_weights = []
    
    graph_info = np.argsort(-1 * full_sim, axis = 1 )

    seen = set()
    
    for i in range(len(meshes)):
        for j in range(1,min(patchnn+1, len(meshes))):
            n = graph_info[i][j]

            pair = (min(i,n), max(i,n))

            if pair in seen:
                continue

            seen.add(pair)

            a = pair[0]
            b = pair[1]
            
            sim = full_sim[a][b]
            edges.append([a, b])
            edge_weights.append(sim)

    edges = np.array(edges).astype(np.int32)
    edge_weights = np.array(edge_weights)
           
    return edges, edge_weights

# meshes are (v,f) for each region
# reg_dist is probability dist n x l
def eval_mrf(meshes, reg_dist, grammar, lmbda):

    unary_cost = -1 * np.log(reg_dist)

    label_cost = lmbda * (grammar.label_dists + 1)
    
    edges, edge_weights = makeMRF(meshes, reg_dist)
    
    preds = mrf_solve(
        unary_cost,
        label_cost,
        edges,
        edge_weights
    )

    return preds


