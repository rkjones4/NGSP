import os, sys, copy
import numpy as np
import networkx as nx
from copy import deepcopy

# Take in a category return a Grammar object

HIER_DATA_PATH = '../hier'

BAD_GEOM_NODE_NAMES = ['storage_furniture/cabinet']
BAD_LAY_NODE_NAMES = ['storage_furniture']

class Grammar:
    def __init__(self, category, granul = '2'):

        hier_folder = f'{HIER_DATA_PATH}/{category}'
        
        self.seen_labels = set()
        self.ornodes = set()
        
        self.start_symbol = None
        self.prod_rules = {}                
        self.parent_map = {}                
        
        self.init_from_file(hier_folder+'/hier.txt')        
        self.fl2i, self.fi2l = self.full_onehot()
        self.hl2i, self.hi2l = self.hier_onehot()
        self.hc_max = max([len(c) for c in self.hl2i.values()])
        
        self.make_par_map()

        self.make_node_to_child_label_maps()
        
        if granul is not None:
            
            self.terminals = self.get_terminals(hier_folder+f'/level-{granul}.txt')
            l2i, i2l = self.leaf_onehot()
                        
            self.l2i = l2i
            self.i2l = i2l
            self.make_node_map()                    
                                
            # list where for each index of a fl2i maps to either a terminal in the current granularity or -1 (if has no parent thats a terminal in this level)
            self.level_map = np.array(self.make_level_map())

            self.make_level_cuts()
            self.make_label_dists()
                                
    
    def get_geom_node_list(self):
        node_list_names = list(self.node_map.keys())
        node_list = [
            self.fl2i[n] for n in node_list_names \
            if n not in BAD_GEOM_NODE_NAMES
        ]
        node_list.sort()
        return node_list

    def get_lay_node_list(self):
        node_list_names = list(self.node_map.keys()) + [self.start_symbol]    
        node_list = [
            self.fl2i[n] for n in node_list_names \
            if ((n not in self.terminals)
                and
                (n not in BAD_LAY_NODE_NAMES)
            )
        ]
        node_list.sort()
        return node_list
        
    def make_node_to_child_label_map(self, l):
        m = np.ones(len(self.fi2l.keys())).astype('long') * -1

        m[self.fl2i[l]] = 0
        
        if l not in self.prod_rules:            
            return m

        for r in self.prod_rules[l]:
            cl = self.hl2i[l][r]
            q = [r]
            while len(q) > 0:
                t = q.pop(0)
                ti = self.fl2i[t]
                m[ti] = cl
                if t in self.prod_rules:
                    for tr in self.prod_rules[t]:
                        q.append(tr)
        
        return m
        
            
    def make_node_to_child_label_maps(self):
        m = {}
        for l in self.fl2i.keys():
            m[self.fl2i[l]] = self.make_node_to_child_label_map(l)

        self.n2cl = m

        
    def make_level_map(self, label_set=None, label_map = None):
        if label_set is None:
            label_set = self.terminals
            label_map = self.l2i
            
        level_map = []
        term_set = set(label_set)
        
        for i,l in self.fi2l.items():
            lab = -1
            
            while l != self.start_symbol:
                if l in term_set:
                    if lab >= 0:
                        assert False
                    lab = label_map[l]                    
                l = self.parent_map[l]

            level_map.append(lab)

        return level_map
        

    def get_terminals(self, level_file):
        terminals = []
        with open(level_file) as f:
            for line in f:
                terminals.append(line.split()[1].strip())

        return terminals
        
            
    def hier_onehot(self):
        hl2i = {}
        hi2l = {}
        for l, r in self.prod_rules.items():
            _l2i = {}
            _i2l = {}
            c = 0
            rhs = copy.deepcopy(r)
            rhs.sort()
            for s in rhs:
                _l2i[s] = c
                _i2l[c] = s
                c += 1
            hl2i[l] = _l2i
            hi2l[l] = _i2l

        return hl2i, hi2l

    def full_onehot(self):
        fl2i = {}
        fi2l = {}
        q = [self.start_symbol]
        c = 0
        while(len(q) > 0):

            l = q.pop(0)
            fl2i[l] = c
            fi2l[c] = l
            c += 1

            if l in self.prod_rules:
                rhs = self.prod_rules[l]
                rhs.sort()
                q += rhs

        return fl2i, fi2l
             
    def leaf_onehot(self):
        l2i = {}
        i2l = {}
        c = 0
        terminals = list(self.terminals)
        terminals.sort()
        for t in terminals:
            l2i[t] = c
            i2l[c] = t
            c += 1
        return l2i, i2l

    
    def make_par_map(self):
        q = [self.start_symbol]
        while len(q) > 0:
            l = q.pop(0)
            rhs = self.prod_rules[l]
            for r in self.prod_rules[l]:
                self.parent_map[r] = l
                if r in self.prod_rules:
                    q.append(r)                
        
    def process_line(self, line):
        if len(line) == 0:
            return
        
        _, name, node_type = line.split()

        if node_type == 'subtypes':
            self.ornodes.add(name)        
        
        if self.start_symbol is None:
            self.start_symbol = name.split('/')[0]

        pn = f'{self.start_symbol}/'
        ln = self.start_symbol
        
        for nn in name.split('/')[1:]:
            n = pn + nn
            if n not in self.seen_labels:
                self.seen_labels.add(n)
                if ln not in self.prod_rules:
                    self.prod_rules[ln] = [n]
                else:
                    self.prod_rules[ln].append(n)
            ln = n
            pn += f'{nn}/'

        
        
    def init_from_file(self, hier_file):
        with open(hier_file) as f:
            for line in f:
                self.process_line(line)            
                    
    def print_grammar(self):
        q = [self.start_symbol]
        while len(q) > 0:
            l = q.pop(0)
            rhs = [r.split('/')[-1] if '/' in r else r for r in self.prod_rules[l] if r in self.node_map]
            print(f'{l} -> ' + ', '.join(rhs))
            for r in self.prod_rules[l]:
                #if r in self.prod_rules:
                if r in self.prod_rules and r not in self.terminals and r in self.node_map:
                    q.append(r)

    # create map from each node -> ([leaf nodes], node type ('top', 'leaf', 'mid') 
    def make_node_map(self):
        node_map = {}

        top_nodes = set(self.prod_rules[self.start_symbol])
        leaf_nodes = self.terminals

        q = [(t, self.l2i[t]) for t in self.terminals]

        while len(q) > 0:
            n, i = q.pop(0)

            if n in node_map:
                node_map[n][0].append(i)

            else:
                
                typ = 'mid'
                        
                if n in top_nodes:
                    typ = 'top'

                if n in leaf_nodes:
                    typ = 'leaf'

                node_map[n] = [[i], typ]

            if self.parent_map[n] != self.start_symbol:
                q.append((self.parent_map[n], i))

        self.node_map = node_map            

    def add_hier_map(self, mode):

        if mode == 'full':
            net_nodes = set([r for r in self.prod_rules.keys() if (r in self.node_map) and (r not in self.terminals)] + [self.start_symbol])
        elif mode == 'twostep':
            net_nodes = set([r for r in self.prod_rules[self.start_symbol] if (r in self.node_map) and (r not in self.terminals)] + [self.start_symbol])
        else:            
            assert False
            
        hier_map = {n:{} for n in net_nodes}

        q = [(self.start_symbol, self.start_symbol)]

        while len(q) > 0:
            h, n = q.pop(0)
                
            for r in self.prod_rules[n]:
        
                if r not in self.node_map:
                    continue

                if r in self.terminals:
                    hier_map[h][r] = len(hier_map[h])
                
                elif r in net_nodes:
                    hier_map[h][r] = len(hier_map[h])
                    q.append((r, r))
                                        
                else:
                    q.append((h, r))
                    
        self.hier_map = hier_map

    def make_level_cuts(self):

        levels = [self.terminals]
        
        cur = self.terminals

        while True:
            work_left = False
            
            nxt = []
            
            for c in cur:
                n = self.parent_map[c]
                
                if n != self.start_symbol: 
                    work_left = True
                    nxt.append(n)
                else:
                    nxt.append(c)
                
            if not work_left:
                break

            rm = set()
                        
            cur = set(nxt)

            for c in cur:
                _c = c
                while _c != self.start_symbol:
                    _c = self.parent_map[_c]
                    if _c in cur:
                        rm.add(c)
                        break
                    
            cur = cur - rm
                    
            levels = [list(cur)] + levels

        cut_levels = []
        for l in levels:
            l.sort()
            lev_l2i = {_l:i for i,_l in enumerate(l)}
            lev_i2l = {i:_l for i,_l in enumerate(l)}
            _level_map = np.array(self.make_level_map(l, lev_l2i))
            cut_levels.append((l, lev_l2i, lev_i2l, _level_map))

        self.cut_levels = cut_levels

    def make_label_dists(self):
        G = nx.Graph()
        G.add_nodes_from(list(range(len(self.fi2l))))

        edges = []

        q = deepcopy(self.terminals)

        while len(q) > 0:
            c = q.pop(0)
            if c not in self.parent_map:
                continue
            n = self.parent_map[c]
            edges.append((self.fl2i[c], self.fl2i[n]))
            edges.append((self.fl2i[n], self.fl2i[c]))
            q.append(n)
            
        G.add_edges_from(edges)

        paths = list(nx.all_pairs_shortest_path(G))

        dists = []

        for t in self.terminals:
            i = self.fl2i[t]
            _dists = []
            for u in self.terminals:
                j = self.fl2i[u]
                d = len(paths[i][1][j]) - 2

                _dists.append(d)
            dists.append(_dists)

        dists = np.array(dists)
        self.label_dists = dists
                         
