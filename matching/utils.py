import pickle
import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import ot
from typing import Optional
from sklearn.neighbors import NearestNeighbors
import sys
sys.path.insert(1, "/mnt/ps/home/CORP/johnny.xi/sandbox/matching/scot/src")
from scotv1 import *

def read_from_pickle(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def label_and_write(predictions, counts, filename):
    """
    Function to use gene_counts to label the predictions and write to pickle file
    """
    pred_dict = {gene:pred_ for (gene, pred_) in zip(counts, predictions)}
    filename = filename + "embeddings.pkl"

    with open(filename, 'wb') as f:
        pickle.dump(pred_dict, f)

def write_to_pickle(object, path):
    with open(path, 'wb') as f:
        pickle.dump(object, f)

def cost_mx_from_probs(prob1, prob2, p = 5, transform_logit = True):
    """
    Compute the cost matrix between two classification probability/logit outputs, using a p-norm
    """
    ### prob1, prob2 are (n x 128) dimensional probabilities 
    prob1, prob2 = prob1.float(), prob2.float()
    if transform_logit:
        prob1, prob2 = torch.nn.functional.log_softmax(prob1, dim = 1), torch.nn.functional.log_softmax(prob2, dim = 1)
    prob1, prob2 = prob1.detach().numpy(), prob2.detach().numpy()
    #dist = np.squeeze(cdist(prob1, prob2, metric = "cosine"))
    dist = np.squeeze(cdist(prob1, prob2, metric = "minkowski", p = p))
    return dist

def hungarian_matching(prob1, prob2, subset = None, **kwargs):
    """
    Compute the optimal matching based on minimizing overall pairwise distances, using the Hungarian algorithm.
    """
    if subset is None:
        subset = [i for i in range(prob1.shape[1])]
    prob1, prob2 = prob1[:, subset], prob2[:, subset]
    cost = cost_mx_from_probs(prob1, prob2, **kwargs)
    #cost += np.random.randn(*cost.shape)*0.0001 ## break ties
    row_reorder, col_reorder = linear_sum_assignment(cost)
    conf_matrix = cost**(-1)
    conf_matrix = conf_matrix/conf_matrix.sum(axis=0)

    return col_reorder, conf_matrix

def snn_matching(x: np.ndarray, y: np.ndarray, k: Optional[int] = 1):

    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)

    ky = k or min(round(0.01 * y.shape[0]), 1000)
    nny = NearestNeighbors(n_neighbors=ky, p=2).fit(y)
    x2y = nny.kneighbors_graph(x)
    y2y = nny.kneighbors_graph(y)

    kx = k or min(round(0.01 * x.shape[0]), 1000)
    nnx = NearestNeighbors(n_neighbors=kx, p=2).fit(x)
    y2x = nnx.kneighbors_graph(y)
    x2x = nnx.kneighbors_graph(x)

    x2y_intersection = x2y @ y2y.T
    y2x_intersection = y2x @ x2x.T
    jaccard = x2y_intersection + y2x_intersection.T
    jaccard.data = jaccard.data / (2 * kx + 2 * ky - jaccard.data)
    matching_matrix = jaccard.multiply(1 / jaccard.sum(axis=1))
    
    return matching_matrix.toarray()

def eot_matching(x: np.ndarray, y: np.ndarray, max_iter = 1000, verbose: bool = True, use_sinkhorn_log = False):
    if use_sinkhorn_log: 
        method = "sinkhorn_log" 
        reg = 0.01
    else: 
        method = "sinkhorn"
        reg = 0.05
    p = ot.unif(x.shape[0])
    q = ot.unif(y.shape[0])
    M = ot.dist(x, y,  metric = "euclidean")
    coupling, log= ot.sinkhorn(p, q, M, reg = reg, numItermax=max_iter, stopThr=1e-9, method = method, log=True, verbose=verbose)
    coupling = coupling/coupling.sum(axis = 1, keepdims = True)

    return coupling

def scot_matching(x: np.ndarray, y: np.ndarray):
    scot = SCOT(x, y)
    _ = scot.align()
    coupling = scot.coupling
    weights=np.sum(coupling, axis = 1)
    coupling = coupling / weights[:, None]

    return coupling

def compute_class_weights(y: np.ndarray) -> np.ndarray:
    assert len(y.shape) == 1,f"Label should be 1-d but got {y.shape}"
    _, counts = np.unique(y, return_counts = True)
    class_probabilities = counts/len(y)
    n_classes = len(class_probabilities)
    return n_classes, 1 / (n_classes * class_probabilities)

def latent_matching_score(coupling: np.ndarray, 
                          z: np.ndarray) -> float:
    z_matched = coupling @ z   ### (n x n) x (n x 4)
    MSE = ((z - z_matched)**2).mean()

    return MSE

def convert_to_labels(y: np.ndarray) -> np.ndarray:
    lookup = {tuple(np.unique(y, axis = 0)[i]):i for i in range(len(np.unique(y, axis = 0)))}
    y_tuple = tuple(map(tuple,y))
    y = np.asarray([lookup[key] for key in y_tuple]) ## (20000,)

    return y
