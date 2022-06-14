from grammar import Grammar
import sys, os, torch
import numpy as np
import argparse
import data_utils, utils, eval_utils
from torch.utils.data import DataLoader
from models import PointNetPPCls
from utils import device
import ast
import random
from copy import deepcopy
from tqdm import tqdm
import train_guide_net as tfs
import train_sem_label_models as ts
import itertools
import sem_label_data_utils as sdu
import math

VERBOSE = False

MIN_LL = -100.
LOG_EPS = 1e-40

class Dataset:
    def __init__(
        self, data, grammar, args
    ):

        self.data = data                                
        self.grammar = grammar
        
        self.samps = []
        self.segments = []
        self.seg_maps = []        

        self.eval_samps = []
        self.eval_segments = []
        self.eval_labels = []
        
        with torch.no_grad():
            for meshes, seg_labels in tqdm(list(zip(self.data['meshes'], self.data['labels']))):
                seg_labels = torch.from_numpy(grammar.level_map[seg_labels]).long().to(device)
                samps, segments, samp_labels = sampleShape(meshes, seg_labels) 
                
                self.eval_samps.append(                    
                    samps[:utils.NUM_EVAL_POINTS].cpu().numpy().astype('float32')
                )
                self.eval_segments.append(segments[:utils.NUM_EVAL_POINTS].cpu().numpy().astype('int16'))
                self.eval_labels.append(samp_labels[:utils.NUM_EVAL_POINTS].cpu().numpy().astype('int16'))
                
                seg_map = []
                for i in range(len(meshes)):
                    si = (segments == i).nonzero().flatten()[:utils.NUM_SEARCH_INPUT_PTS].long().cpu()
                    seg_map.append(si)

                exp_inds = torch.zeros(segments.shape, device=device)
                seg_inds = torch.cat(seg_map,dim=0)
                exp_inds[seg_inds] = 1
                small_inds = exp_inds.nonzero().flatten()
                
                small_samps = samps[small_inds]
                small_segments = segments[small_inds]
                
                self.samps.append(
                    small_samps.cpu().numpy().astype('float32')
                )
                self.segments.append(small_segments.cpu().numpy().astype('int16'))

                s2i_map = {}
                for i in range(len(meshes)):
                    si = (small_segments == i).nonzero().flatten()

                    if si.shape[0] == 0:
                        continue
                    
                    si = si.repeat((utils.NUM_SEARCH_INPUT_PTS//si.shape[0])+1)
                    si = si[:utils.NUM_SEARCH_INPUT_PTS].long()                    
                    s2i_map[i] = si

                self.seg_maps.append(s2i_map)

                
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
        faces, verts.unsqueeze(0), utils.NUM_SAMP_PTS
    )
        
    samp_segments = segments[face_inds]    
    samp_labels = labels[face_inds]
        
    return samps, samp_segments, samp_labels

