import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pygco


# unary_cost -> n x l
# label_cost -> l x l
# edges -> M x 2
# edge_weights -> M
def mrf_solve(unary_cost, label_cost, edges, edge_weights):
    init_labels = unary_cost.argmin(axis=1)

    if len(edges) == 0:
        return init_labels
    
    return pygco.cut_general_graph(
        edges,
        edge_weights,
        unary_cost,
        label_cost,
        n_iter=-1,
        algorithm="expansion",
        init_labels = init_labels        
    )

def main1():
    
    unary_cost = np.array(
        [[0.0, 1.0, 2.0],
         [4.0, 1.0, 0.0],
         [1.0, 0.0, 2.0]]
    )
    
    label_cost = (1 - np.eye(3)).astype(np.float)
    
    edges = np.array(
        [[0, 1],
         [1, 2],
         [0, 2]]
    ).astype(np.int32)
    
    edge_weights = np.array([2.0, 0.0, 0.0])

    print(mrf_solve(unary_cost, label_cost, edges, edge_weights))

def main2():
    
    unary_cost = np.array(
        [[0.3, 0.5, 2.0],
         [2.0, 0.4, 0.2],
         [0.1, 0.2, 0.1]]
    )
    
    label_cost = (1 - np.eye(3)).astype(np.float)
    
    edges = np.array(
        [[0, 1],
         [1, 2],
         [0, 2]]
    ).astype(np.int32)
    
    edge_weights = np.array([.01, 0.01, 0.01])

    print(mrf_solve(unary_cost, label_cost, edges, edge_weights))
    
if __name__ == "__main__":
    main2()
