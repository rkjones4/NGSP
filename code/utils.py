import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import argparse
import sys
import ast

NUM_SAMP_PTS = 1000000

NUM_PROP_INPUT_PTS = 10000
NUM_SEARCH_INPUT_PTS = 4096

NUM_EVAL_POINTS = 10000

device = torch.device('cuda')

DEF_ARGS = [    
    # Set per run
    ('-en', '--exp_name', None,  str),            
    ('-c', '--category', None,  str),
    
    # Have defaults
    ('-ts', '--train_size', 400, int),    
    ('-ubn', '--use_bn', "False", str),
    ('-es', '--eval_size', 10000, int),
    ('-tn', '--train_name', 'train', str),    
    ('-dp', '--data_path', '../data',  str),
    ('-rd', '--rd_seed', 42,  int),
    ('-b', '--batch_size', 16,  int),
    ('-ep', '--epochs', 10000,  int),    
    ('-lr', '--lr', 1e-3,  float),
    ('-o', '--outpath', 'model_output',  str),
    ('-prp', '--print_per', 5,  int),
    ('-evp', '--eval_per', 5,  int),    
    ('-esp', '--es_patience', 50,  int),
    ('-est', '--es_threshold', 0.001,  float),
    ('-esm', '--es_metric', 'mIoU',  str),
    ('-evo', '--eval_only', 'n', str),
    ('-emp', '--eval_model_path', None, str),    
]

SEARCH_DEF_ARGS = [        
    ('-mpn', '--max_prop_negs', 100,  int),
    ('-minen', '--min_eval_negs', 10,  int),
    ('-d', '--dropout', 0.2, float),
    ('-sw', '--scale_weight', 0.05, float),
    ('-nw', '--noise_weight', 0.005, float),
    ('-lr', '--lr', 0.0001, float),
    ('-est', '--es_threshold', 0.001,  float),
    ('-esm', '--es_metric', 'Mean_Accuracy',  str), 
    ('-esp', '--es_patience', 100,  int),
    ('-mnas', '--max_neg_area_sim', 0.95, float),
    ('-nl', '--node_list', None, str),
    ('-sdp','--search_data_path', None, str),
    ('-evo', '--eval_only', 'n', str),
    ('-emp', '--eval_model_path', None, str),    
]
    

EVAL_PROP_LOG_INFO = [    
    ('mIoU', 'all_iou', 'nc'),
    ('Accuracy', 'corr', 'total'),
    ('Avg LL', 'll', 'count'),
]

EVAL_SAMPLE_LOG_INFO = [
    ('Seg Target Perc', 'seg_tar', 'nc'), # % target was in the samples
    ('Seg Best Acc', 'seg_best_acc', 'nc'), # average best accuracy out of all samples
    ('Seg First Acc', 'seg_first_acc', 'nc'), # average accuracy of first sample
    ('Pt Best Acc', 'pt_best_acc', 'nc'), # average best accuracy out of all samples
    ('Pt First Acc', 'pt_first_acc', 'nc'), # average accuracy of first sample
    ('Best mIoU', 'best_iou', 'nc'), # mIoU using best accuracy samples
    ('First mIoU', 'first_iou', 'nc') # mIoU using first accuracy samples
]

SEARCH_TRAIN_LOG_INFO = [
    ('Loss', 'loss', 'batch_count'),
    ('Accuracy', 'corr', 'total'),
    ('Pos Accuracy', 'pos_corr', 'pos_total'),
    ('Neg Accuracy', 'neg_corr', 'neg_total'),
    ('Pos Loss', 'pos_loss', 'pos_total'),
    ('Neg Loss', 'neg_loss', 'neg_total'),
    ('Mean Accuracy', 'pos_corr', 'pos_total', 'neg_corr', 'neg_total'),
    ('Mean Loss', 'pos_loss', 'pos_total', 'neg_loss', 'neg_total'),
]

SEARCH_TRAIN_LOG_INFO = [
    ('Loss', 'loss', 'nc'),
    ('Accuracy', 'corr', 'total'),
    ('Pos Accuracy', 'pos_corr', 'pos_total'),
    ('Neg Accuracy', 'neg_corr', 'neg_total'),
]

SEARCH_EVAL_LOG_INFO = [
    ('Mean_Accuracy', 'mean_acc', 'nc'),
    ('Accuracy', 'acc', 'nc'),    
    ('False Pos', 'fp', 'nc'),
    ('False Neg', 'fn', 'nc'),
    ('Prob Margin', 'prob_margin', 'nc'),
    ('Pos First', 'pos_first', 'nc'),
    ('Pos_Better', 'pos_better', 'nc'),
    ('Weighted_Metric', 'wm', 'nc'),
    ('NegLoss', 'neg_loss', 'nc'),
]

PROP_TRAIN_LOG_INFO = [
    ('Loss', 'loss', 'batch_count'),
    ('Accuracy', 'corr', 'total')
]

def getArgs(arg_list):       
    def_args =  DEF_ARGS

    parser = argparse.ArgumentParser()

    seen = set()
    
    for s,l,d,t in arg_list:        
        parser.add_argument(s, l, default=d, type = t)
        seen.add(s)
        seen.add(l)
        
    for s,l,d,t in def_args:
        if s not in seen and l not in seen:
            parser.add_argument(s, l, default=d, type = t)
        else:
            print(f"Skipping {s}:{l} default arg")

    args = parser.parse_args()
    args.use_bn = ast.literal_eval(args.use_bn)
        
    return args
    
def writeSPC(pc, fn):
    with open(fn, 'w') as f:
        for a, b, c in pc:
            f.write(f'v {a} {b} {c} \n')

def init_model_run(args, model_type=None):
        
    random.seed(args.rd_seed)
    np.random.seed(args.rd_seed)
    torch.manual_seed(args.rd_seed)

    os.system(f'mkdir {args.outpath} > /dev/null 2>&1')
    
    os.system(f'mkdir {args.outpath}/{args.exp_name} > /dev/null 2>&1')

    if model_type is not None:
        args.exp_name += f'/{model_type}'
        os.system(f'mkdir {args.outpath}/{args.exp_name} > /dev/null 2>&1')

    with open(f"{args.outpath}/{args.exp_name}/config.txt", "w") as f:
        f.write(f'CMD: {" ".join(sys.argv)}\n')        
        f.write(f"ARGS: {args}\n")            

    if model_type is None:
        print("Warning: No Model Type")
        return
        
    os.system(f'mkdir {args.outpath}/{args.exp_name}/plots > /dev/null 2>&1')
    os.system(f'mkdir {args.outpath}/{args.exp_name}/plots/train > /dev/null 2>&1')
    os.system(f'mkdir {args.outpath}/{args.exp_name}/plots/eval > /dev/null 2>&1')        
    os.system(f'mkdir {args.outpath}/{args.exp_name}/part_seg > /dev/null 2>&1')
    os.system(f'mkdir {args.outpath}/{args.exp_name}/part_seg/train > /dev/null 2>&1')
    os.system(f'mkdir {args.outpath}/{args.exp_name}/part_seg/val > /dev/null 2>&1')
    os.system(f'mkdir {args.outpath}/{args.exp_name}/part_seg/test > /dev/null 2>&1')
    os.system(f'mkdir {args.outpath}/{args.exp_name}/pc_seg > /dev/null 2>&1')
    os.system(f'mkdir {args.outpath}/{args.exp_name}/pc_seg/train > /dev/null 2>&1')
    os.system(f'mkdir {args.outpath}/{args.exp_name}/pc_seg/val > /dev/null 2>&1')
    os.system(f'mkdir {args.outpath}/{args.exp_name}/pc_seg/test > /dev/null 2>&1')    
    os.system(f'mkdir {args.outpath}/{args.exp_name}/models > /dev/null 2>&1')
    
def loadColPC(infile):
    verts = []
    colors = []
    with open(infile) as f:
        for line in f:
            ls = line.split()
            if len(ls) == 0:
                continue
            if ls[0] == 'v':
                verts.append([
                    float(ls[1]),
                    float(ls[2]),
                    float(ls[3])
                ])
                colors.append([
                    float(ls[4]),
                    float(ls[5]),
                    float(ls[6]),
                ])
                
    return verts, colors
                
                
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

def loadAndCleanObj(infile):
    raw_verts = []
    raw_faces = []
    seen_verts = set()
    with open(infile) as f:
        for line in f:
            ls = line.split()
            if ls[0] == 'v':
                raw_verts.append((
                    float(ls[1]),
                    float(ls[2]),
                    float(ls[3])
                ))
            elif ls[0] == 'f':
                ls = [i.split('//')[0] for i in ls]
                raw_faces.append((
                    int(ls[1]),
                    int(ls[2]),
                    int(ls[3]),
                ))
                seen_verts.add(int(ls[1]) -1)
                seen_verts.add(int(ls[2]) -1)
                seen_verts.add(int(ls[3]) -1)

    seen_verts = list(seen_verts)
    seen_verts.sort()
    sv_map = {}
    for i,vn in enumerate(seen_verts):
        sv_map[vn] = i + 1
        
    seen_verts = set(seen_verts)

    verts = []
    faces = []
    for i, v in enumerate(raw_verts):
        if i in seen_verts:
            verts.append(v)

    for face in raw_faces:
        faces.append(
            (
                sv_map[face[0]-1] -1,
                sv_map[face[1]-1] -1,
                sv_map[face[2]-1] -1,
            )
        )

    verts = np.array(verts).astype('float16')
    faces = np.array(faces).astype('long')
    
    return verts, faces

def face_areas_normals(faces, vs):
    face_normals = torch.cross(
        vs[:, faces[:, 1], :] - vs[:, faces[:, 0], :],
        vs[:, faces[:, 2], :] - vs[:, faces[:, 1], :],
        dim=2,
    )
    face_areas = torch.norm(face_normals, dim=2) + 1e-8
    face_normals = face_normals / face_areas[:, :, None]
    face_areas = 0.5 * face_areas
    return face_areas, face_normals


def sample_surface(faces, vs, count):
    """
    sample mesh surface
    sample method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    Args
    ---------
    vs: vertices (batch x nvs x 3d coordinate)
    faces: triangle faces (torch.long) (num_faces x 3)
    count: number of samples
    Return
    ---------
    samples: (count, 3) points in space on the surface of mesh
    face_index: (count,) indices of faces for each sampled point
    """
    
    if torch.isnan(faces).any() or torch.isnan(vs).any():
        assert False, 'saw nan in sample_surface'

    device = vs.device
    bsize, nvs, _ = vs.shape
    area, normal = face_areas_normals(faces, vs)
    area_sum = torch.sum(area, dim=1)

    assert not (area <= 0.0).any().item(
    ), "Saw negative probability while sampling"
    assert not (area_sum <= 0.0).any().item(
    ), "Saw negative probability while sampling"
    assert not (area > 1000000.0).any().item(), "Saw inf"
    assert not (area_sum > 1000000.0).any().item(), "Saw inf"

    dist = torch.distributions.categorical.Categorical(
        probs=area / (area_sum[:, None]))
    
    face_index = dist.sample((count,))
    keep_face_index = face_index.clone()
    
    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vs[:, faces[:, 0], :]
    tri_vectors = vs[:, faces[:, 1:], :].clone()
    tri_vectors -= tri_origins.repeat(
        1,
        1,
        2
    ).reshape((bsize, len(faces), 2, 3))

    # pull the vectors for the faces we are going to sample from
    face_index = face_index.transpose(0, 1)
    face_index = face_index[:, :, None].expand((bsize, count, 3))
    tri_origins = torch.gather(tri_origins, dim=1, index=face_index)
    face_index2 = face_index[:, :, None, :].expand((bsize, count, 2, 3))
    tri_vectors = torch.gather(tri_vectors, dim=1, index=face_index2)

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = torch.rand(
        count, 2, 1, device=vs.device, dtype=tri_vectors.dtype)

    # points will be distributed on a quadrilateral if we use 2x [0-1] samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(dim=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = torch.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths[None, :]).sum(dim=2)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    normals = torch.gather(normal, dim=1, index=face_index)[0]
    
    return samples[0], keep_face_index.squeeze(), normals

def log_print(s, args, fn='log'):
    of = f"{args.outpath}/{args.exp_name}/{fn}.txt"
    with open(of, 'a') as f:
        f.write(f"{s}\n")
    print(s)
    
def print_results(
    LOG_INFO,
    result,
    args
):
    res = ""
    for info in LOG_INFO:
        if len(info) == 3:
            name, key, norm_key = info
            if key not in result:
                continue
            _res = result[key] / (result[norm_key]+1e-8)
                
        elif len(info) == 5:
            name, key1, norm_key1, key2, norm_key2 = info
            if key1 not in result or key2 not in result:
                continue
            res1 = result[key1] / (result[norm_key1]+1e-8)
            res2 = result[key2] / (result[norm_key2]+1e-8)
            _res = (res1 + res2) / 2
                
        else:
            assert False, f'bad log info {info}'
                                     
        res += f"    {name} : {round(_res, 4)}\n"

    log_print(res, args)

def make_plots(
    LOG_INFO,
    results,
    plots,
    epochs,
    args,
    fname
):
    for info in LOG_INFO:
        
        for rname, result in results.items():
            if len(info) == 3:
                name, key, norm_key = info
                if key not in result:
                    continue
                res = result[key] / (result[norm_key]+1e-8)
                
            elif len(info) == 5:
                name, key1, norm_key1, key2, norm_key2 = info
                if key1 not in result or key2 not in result:
                    continue
                res1 = result[key1] / (result[norm_key1]+1e-8)
                res2 = result[key2] / (result[norm_key2]+1e-8)
                res = (res1 + res2) / 2
                
            else:
                assert False, f'bad log info {info}'
                        
            if name not in plots[rname]:
                plots[rname][name] = [res]
            else:
                plots[rname][name].append(res)



        plt.clf()
        for key in plots:
            if name not in plots[key]:
                continue
            plt.plot(
                epochs,
                plots[key][name],
                label= key
            )
        plt.legend()
        plt.grid()
        plt.savefig(f'{args.outpath}/{args.exp_name}/plots/{fname}/{name}.png')


def writeObj(verts, faces, outfile):
    with open(outfile, 'w') as f:
        for a, b, c in verts.tolist():
            f.write(f'v {a} {b} {c}\n')

        for a, b, c in faces.tolist():
            f.write(f"f {a+1} {b+1} {c+1}\n")

class SuppressStream(object): 

    def __init__(self, stream=sys.stderr):
        self.orig_stream_fileno = stream.fileno()

    def __enter__(self):
        self.orig_stream_dup = os.dup(self.orig_stream_fileno)
        self.devnull = open(os.devnull, 'w')
        os.dup2(self.devnull.fileno(), self.orig_stream_fileno)

    def __exit__(self, type, value, traceback):
        os.close(self.orig_stream_fileno)
        os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
        os.close(self.orig_stream_dup)
        self.devnull.close()
