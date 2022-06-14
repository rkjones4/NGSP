import torch
import numpy as np
import json
import utils
import os
from make_areas import get_area

from tqdm import tqdm

colors = [
    (31, 119, 180),
    (174, 199, 232),
    (255,127,14),
    (255, 187, 120),
    (44,160,44),
    (152,223,138),
    (214,39,40),
    (255,152,150),
    (148, 103, 189),
    (192,176,213),
    (140,86,75),
    (196,156,148),
    (227,119,194),
    (247,182,210),
    (127,127,127),
    (199,199,199),
    (188,188,34),
    (219,219,141),
    (23,190,207),
    (158,218,229)
]

color_names = [
    'royal',
    'sky',
    'orange',
    'goldenrod',
    'forest',
    'lime',
    'red',
    'strawberry',
    'purple',
    'violet',
    'brown',
    'cafe',
    'fuschia',
    'bubblegum',
    'dgrey',
    'lgrey',
    'rio',
    'yellow',
    'aqua',
    'baby'
]

def get_color_name(i):
    ri = i % len(colors)
    num_over = (i // len(colors))
    over = ((num_over + 1) // 2) * 55
    
    sign = 2 * (((num_over+1) % 2 == 0) - .5)    
    delta = over * sign
    raw_name = color_names[ri]

    
    if delta == 0:
        pre = ''
    elif delta < 0:
        pre = 'light '
    elif delta > 0:
        pre = 'dark '

    return pre + raw_name
        
def color_info(grammar):
    for i,l in grammar.i2l.items():
        n = get_color_name(i)
        print(f'{n} : {l.replace("/"," ")}')
        
def get_color(i):
    ri = i % len(colors)
    num_over = (i // len(colors))
    over = ((num_over + 1) // 2) * 55
    
    sign = 2 * (((num_over+1) % 2 == 0) - .5)    
    delta = over * sign
    raw_color = colors[ri]    
    return tuple([
        min(max(c+delta,0),255)
        for c in raw_color
    ])

def save_part_data(dir_name, parts, grammar):
    os.system(f'mkdir {dir_name} > /dev/null 2>&1')
    with open(f'{dir_name}/info.txt', 'w') as fil:
        for i, (v, f, l) in enumerate(parts):
            ll = l
            name = grammar.i2l[ll]
            assert name in grammar.terminals
            fil.write(f'part {i} label {ll} name {name}\n')
            utils.writeObj(v,f,f'{dir_name}/part_{i}.obj')

            
def save_pc_data(dir_name, samps, segments, seg_labels, grammar):
    os.system(f'mkdir {dir_name} > /dev/null 2>&1')
    with open(f'{dir_name}/info.txt', 'w') as fil:
        assert samps.shape[0] == segments.shape[0]
        segs = segments.unique().tolist()
        segs.sort()
        for i in segs:
            seg_inds = (segments == i).nonzero().view(-1)
            points = samps[seg_inds]
            l = seg_labels[i]
            name = grammar.i2l[l]
            assert name in grammar.terminals
            fil.write(f'part {i} label {l} name {name}\n')
            np.save(f'{dir_name}/part_{i}.npy', points)



def vis_parts(mesh_name, parts):
    face_offset = 1
    
    o_verts = []
    o_faces = []

    for (verts, faces, l) in parts:
        _fo = 0


        color = get_color(l)
        for a, b, c in verts:
            o_verts.append(f'v {a} {b} {c} {color[0]} {color[1]} {color[2]}\n')
            _fo += 1

        for a, b, c in faces:
            o_faces.append(f'f {a+face_offset} {b+face_offset} {c+face_offset}\n')
            
        face_offset += _fo

    with open(mesh_name, 'w') as f:
        for v in o_verts:
            f.write(v)
            
        for fa in o_faces:
            f.write(fa)

def vis_pc(pc_name, points, labels, write_img=False):
    pc_verts = []

    img_pts = []
    img_colors = []
    
    if points is np.ndarray:
        points = torch.from_np(points)
    if labels is np.ndarray:
        labels = torch.from_np(labels)
        
    for i in labels.cpu().unique().flatten():
        inds = (labels == i).squeeze().nonzero().squeeze()
        color = get_color(i.item())
        pts = points[inds].view(-1, 3)
        for a,b,c in pts:
            pc_verts.append(f'v {a} {b} {c} {color[0]} {color[1]} {color[2]}\n')
            if write_img:
                img_pts.append([a,b,c])
                img_colors.append(color)
            
    with open(pc_name, 'w') as f:
        for v in pc_verts:
            f.write(v)
        
def make_gt_vis(args, name, data):
    with torch.no_grad():
    
        for ind, parts in zip(data['inds'], data['parts']):
            vis_parts(
                f'{args.outpath}/{args.exp_name}/part_seg/{name}/{ind}_gt.obj',
                parts, 
            )
    
        for ind, samps, labels in zip(
            data['inds'], data['samps'], data['labels']
        ):
            vis_pc(
                f'{args.outpath}/{args.exp_name}/pc_seg/{name}/{ind}_gt.obj',
                samps,
                labels
            )


def load_data(
    path,
    ind,
):
    
    j = json.load(open(f'{path}/{ind}/data.json'))
    labels = np.array(j['labels']).astype('long')
    parts = j['parts']

    meshes = []
    for p in parts:
        meshes.append(utils.loadAndCleanObj(p))

    areas = np.load(f'../data/area/area_{ind}.npy')
        
    d = {
        'meshes': meshes,
        'labels': labels,
        'areas': areas
    }
    
    return d

def load_dataset(
    path,
    split_file,
    split_name,    
):
    data = {
        'meshes': [],
        'labels': [],
        'inds': [],
        'areas': []
    }

    split_info = json.load(open(f'data_splits/{split_file}'))
    
    if '_' in split_name:
        sname, num = split_name.split('_')
        num = int(num)            
        inds = split_info[sname][:num]
    else:
        inds = split_info[split_name]
        
    for ind in tqdm(inds):
        with torch.no_grad():
            d = load_data(
                path,
                ind,
            )

        for k,v in d.items():
            data[k].append(v)

        data['inds'].append(ind)
            
    return data

def load_unsup_dataset(
    path,
    split_file,
    train_num,
    unsup_num
):
    data = {
        'meshes': [],
        'labels': [],
        'inds': [],
        'areas': []
    }

    split_info = json.load(open(f'data_splits/{split_file}'))

    inds = []

    for fn in os.listdir(path):
        try:
            inds.append(int(fn))

        except Exception:
            pass


    inds = set(inds)
    inds = inds - set([int(s) for s in split_info['train'][:train_num]])
    inds = inds - set([int(s) for s in split_info['val']])
    inds = inds - set([int(s) for s in split_info['test']])

    inds = list(inds)
    inds.sort()

    inds = inds[:unsup_num]    
    
    for ind in tqdm(inds):
        with torch.no_grad():
            d = load_data(
                path,
                ind,
            )

        for k,v in d.items():
            data[k].append(v)

        data['inds'].append(ind)
            
    return data
