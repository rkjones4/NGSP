# Code from https://github.com/czq142857/BAE-NET

import torch
import sys
import os
import numpy as np
import random
from tqdm import tqdm

V_DIM = 128
BATCH_SIZE = 8192

CACHE_DIR = 'cache_baenet'
os.system(f'mkdir {CACHE_DIR}')

def loadObj(infile):
    tverts = []
    ttris = []
    with open(infile) as f:
        for line in f:
            ls = line.split()
            if len(ls) == 0:
                continue
            if ls[0] == 'v':
                tverts.append([
                    float(ls[1]),
                    float(ls[2]),
                    float(ls[3])
                ])
            elif ls[0] == 'f':
                ttris.append([
                    int(ls[1].split('//')[0])-1,
                    int(ls[2].split('//')[0])-1,
                    int(ls[3].split('//')[0])-1
                ])

    return tverts, ttris

def writeObj(verts, faces, outfile):
    with open(outfile, 'w') as f:
        for a, b, c in verts:
            f.write(f'v {a} {b} {c}\n')
        for a, b, c in faces:
            f.write(f"f {a+1} {b+1} {c+1}\n")

def writeSPC(pc, fn):
    with open(fn, 'w') as f:
        for a,b,c in pc:
            f.write(f'v {a} {b} {c} \n')

def readCSV(name):
    a = []
    with open(name) as f:
        for line in f:
            a.append([float(_a) for _a in line[:-1].split(',')])
    return a

# mesh_file -> filename, query_points -> torch.tensor (Nx3)
# returns 1 if inside, 0 if outside
def testInsideOutside(mesh_file, ind, fn='tmp/query_points.csv'):    
    os.system(f'./fastwinding  {mesh_file} {fn} tmp/fwn_res_{ind}.csv')
    fwn_res = torch.tensor(readCSV(f'tmp/fwn_res_{ind}.csv'))
    os.system(f'rm tmp/fwn_res_{ind}.csv')
    
    return (fwn_res >= 0.5).float()

def calc_info(infile, ind):

    flat_voxels = testInsideOutside(infile, ind)    

    voxel_model_dense = (flat_voxels.view(V_DIM, V_DIM, V_DIM).T).numpy()
    
    dim_voxel = 64
    voxel_model_64 = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
    multiplier = int(V_DIM/dim_voxel)
    for i in range(dim_voxel):
        for j in range(dim_voxel):
            for k in range(dim_voxel):
                voxel_model_64[i,j,k] = np.max(
                    voxel_model_dense[
                        i*multiplier:(i+1)*multiplier,
                        j*multiplier:(j+1)*multiplier,
                        k*multiplier:(k+1)*multiplier
                    ])

    dim_voxel = 32
    voxel_model_32 = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
    multiplier = int(V_DIM/dim_voxel)
    for i in range(dim_voxel):
        for j in range(dim_voxel):
            for k in range(dim_voxel):
                voxel_model_32[i,j,k] = np.max(
                    voxel_model_dense[
                        i*multiplier:(i+1)*multiplier,
                        j*multiplier:(j+1)*multiplier,
                        k*multiplier:(k+1)*multiplier
                    ]
                )

    
    batch_size = BATCH_SIZE
    
    sample_points = np.zeros([batch_size,3],np.uint8)
    sample_values = np.zeros([batch_size,1],np.uint8)

    batch_size_counter = 0
    voxel_model_32_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
    for i in range(2,dim_voxel-2):
        if (batch_size_counter>=batch_size): break
        for j in range(2,dim_voxel-2):
            if (batch_size_counter>=batch_size): break
            for k in range(2,dim_voxel-2):
                if (batch_size_counter>=batch_size): break
                if (np.max(voxel_model_32[i-2:i+3,j-2:j+3,k-2:k+3])!=np.min(voxel_model_32[i-2:i+3,j-2:j+3,k-2:k+3])):
                    sample_points[batch_size_counter,0] = i
                    sample_points[batch_size_counter,1] = j
                    sample_points[batch_size_counter,2] = k
                    sample_values[batch_size_counter,0] = voxel_model_32[i,j,k]
                    voxel_model_32_flag[i,j,k] = 1
                    batch_size_counter +=1

    if (batch_size_counter>=batch_size):
        batch_size_counter = 0
        voxel_model_32_flag = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
        for i in range(0,dim_voxel,2):
            for j in range(0,dim_voxel,2):
                for k in range(0,dim_voxel,2):
                    filled_flag = False
                    for (i0,j0,k0) in [(i,j,k),(i+1,j,k),(i,j+1,k),(i+1,j+1,k),(i,j,k+1),(i+1,j,k+1),(i,j+1,k+1),(i+1,j+1,k+1)]:
                        if voxel_model_32[i0,j0,k0]>0:
                            filled_flag = True
                            sample_points[batch_size_counter,0] = i0
                            sample_points[batch_size_counter,1] = j0
                            sample_points[batch_size_counter,2] = k0
                            sample_values[batch_size_counter,0] = voxel_model_32[i0,j0,k0]
                            voxel_model_32_flag[i0,j0,k0] = 1
                            break
                    if not filled_flag:
                        sample_points[batch_size_counter,0] = i
                        sample_points[batch_size_counter,1] = j
                        sample_points[batch_size_counter,2] = k
                        sample_values[batch_size_counter,0] = voxel_model_32[i,j,k]
                        voxel_model_32_flag[i,j,k] = 1
                    batch_size_counter +=1
                        
    while (batch_size_counter<batch_size):
        while True:
            i = random.randint(0,dim_voxel-1)
            j = random.randint(0,dim_voxel-1)
            k = random.randint(0,dim_voxel-1)
            if voxel_model_32_flag[i,j,k] != 1: break
        sample_points[batch_size_counter,0] = i
        sample_points[batch_size_counter,1] = j
        sample_points[batch_size_counter,2] = k
        sample_values[batch_size_counter,0] = voxel_model_32[i,j,k]
        voxel_model_32_flag[i,j,k] = 1
        batch_size_counter +=1
                        
    sample_points = ((sample_points.astype(float) + .5) / 16) - 1.

    sample_values = sample_values.astype(float)
    voxels = voxel_model_64.astype(float)
    
    # debug through visualization
    """    
    writeSPC(sample_points[sample_values.nonzero()[0]], f'test.obj')            

    voxels = torch.from_numpy(voxels)    
    pos_voxel_pts = (((voxels == 1.).nonzero().float() + .5) / 16) - 1.
    neg_voxel_pts = (((voxels == 0.).nonzero().float() + .5) / 16) - 1.

    writeSPC(pos_voxel_pts, 'pos_voxels.obj')
    writeSPC(neg_voxel_pts, 'neg_voxels.obj')
    """
    
    return sample_points, sample_values, voxels
    
def gen_data(verts, faces, ind):
    writeObj(verts, faces, f'tmp/tmp_mesh_{ind}.obj')
    os.system(f'./manifold  tmp/tmp_mesh_{ind}.obj  tmp/tmp_man_mesh_{ind}.obj  > /dev/null 2>&1')
    points, values, voxels = calc_info(f'tmp/tmp_man_mesh_{ind}.obj', ind)

    os.system(f'rm tmp/tmp_mesh_{ind}.obj > /dev/null 2>&1')
    os.system(f'rm tmp/tmp_man_mesh_{ind}.obj > /dev/null 2>&1')

    return points, values, voxels

def load_baenet_data(ind, verts, faces):
    fls = os.listdir(CACHE_DIR)

    if f'{ind}.npz' in fls:
        res = np.load(f'{CACHE_DIR}/{ind}.npz')
        return res['points'], res['values'], res['voxels']
    else:
        points, values, voxels = gen_data(verts, faces, ind)
        np.savez(
            f'{CACHE_DIR}/{ind}.npz',
            points=points,
            values=values,
            voxels=voxels
        )
        return points, values, voxels

if __name__ == '__main__':
    with torch.no_grad():
        verts, faces = loadObj(sys.argv[1])
        gen_data(verts, faces)            
