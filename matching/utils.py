import pickle
import torch
import numpy as np
import pytorch_lightning as pl
import ot
from typing import Callable, Optional, Union, Any, Tuple
from sklearn.neighbors import NearestNeighbors
import sys
from .models.classifier import BallsClassifier, GEXADT_Classifier
from .models.vae import ImageVAEModule, GEXADTVAEModule

sys.path.insert(1, "/mnt/ps/home/CORP/johnny.xi/sandbox/matching/scot/src")
from scotv1 import *
from evals import *
from timeit import default_timer as timer


def nullable_string(val):
    if not val:
        return None
    return val

def read_from_pickle(path: str) -> Any:
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def write_to_pickle(object, path):
    with open(path, 'wb') as f:
        pickle.dump(object, f)

def snn_matching(x: Union[torch.Tensor, np.ndarray], 
                 y: Union[torch.Tensor, np.ndarray], 
                 k: Optional[int] = 1) -> torch.Tensor:

    if isinstance(x, torch.Tensor): x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor): y = y.cpu().detach().numpy()

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
    
    return torch.from_numpy(matching_matrix.toarray())

def eot_matching(x: Union[torch.Tensor, np.ndarray], 
                 y: Union[torch.Tensor, np.ndarray], 
                 max_iter: int = 1000, 
                 verbose: bool = False, 
                 use_sinkhorn_log: bool = True) -> torch.Tensor:
    if use_sinkhorn_log: 
        method = "sinkhorn_log" 
        reg = 0.05
    else: 
        method = "sinkhorn"
        reg = 0.05
    if isinstance(x, np.ndarray): x = torch.from_numpy(x)
    if isinstance(y, np.ndarray): y = torch.from_numpy(y)
    p = ot.unif(x.shape[0], type_as = x).to("cuda")
    q = ot.unif(y.shape[0], type_as = y).to("cuda")
    M = ot.dist(x, y,  metric = "euclidean").to("cuda")
    coupling, log= ot.sinkhorn(p, q, M, reg = reg, numItermax=max_iter, stopThr=1e-10, method = method, log=True, verbose=verbose)
    while torch.isnan(coupling).any() and reg < 0.1:
        reg += 0.01
        coupling, log= ot.sinkhorn(p, q, M, reg = reg, numItermax=max_iter, stopThr=1e-10, method = method, log=True, verbose=verbose)
    coupling = coupling/coupling.sum(dim = 1, keepdims = True)
    return coupling

def scot_matching(x: Union[torch.Tensor, np.ndarray], 
                  y: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(x, torch.Tensor): x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor): y = y.cpu().detach().numpy()

    scot = SCOT(x, y)
    e = 0.01
    _ = scot.align(metric = "correlation", e = e)
    coupling = scot.coupling
    while np.isnan(coupling).any() and e < 0.1:
        e += 0.01
        _ = scot.align(metric = "correlation", e = e)
        coupling = scot.coupling
    weights = np.sum(coupling, axis = 1)
    coupling = coupling / weights[:, None]

    return torch.from_numpy(coupling)

def latent_matching_score(coupling: torch.Tensor, 
                          z: Union[np.ndarray, torch.Tensor]) -> float:
    if isinstance(z, np.ndarray): z = torch.from_numpy(z).to("cuda")
    z_matched = coupling @ z  
    MSE = ((z - z_matched)**2).mean()

    return MSE

def convert_to_labels(y: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(y, torch.Tensor): y = y.cpu().detach().numpy()
    lookup = {tuple(np.unique(y, axis = 0)[i]):i for i in range(len(np.unique(y, axis = 0)))}
    y_tuple = tuple(map(tuple,y))
    y = np.asarray([lookup[key] for key in y_tuple]) 
    return torch.from_numpy(y)

def compute_avg_FOSCTTM(x: Union[np.ndarray, torch.Tensor], 
                        y: Union[np.ndarray, torch.Tensor]) -> float:
    
    if isinstance(x, torch.Tensor): x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor): y = y.cpu().detach().numpy()
    
    return np.array(calc_domainAveraged_FOSCTTM(x, y)).mean()

def load_from_checkpoint_(path: str, dataset: str, device: str = "cuda") -> pl.LightningModule:
    assert device in ["cuda", "cpu"]
    assert dataset in ["BALLS", "GEXADT"]

    if dataset == "BALLS": module_classifier, module_vae = BallsClassifier, ImageVAEModule
    if dataset == "GEXADT": module_classifier, module_vae = GEXADT_Classifier, GEXADTVAEModule

    try:
        model = module_classifier.load_from_checkpoint(path, map_location=torch.device(device))
    except KeyError:
        try:
            model = module_vae.load_from_checkpoint(path, map_location=torch.device(device))
        except KeyError:
            print(f"Checkpoint at {path} did not correspond to a model for dataset {dataset}!")
    
    return model 


