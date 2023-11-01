import sys
import os
import numpy as np
from matching.utils import scot_matching, snn_matching, eot_matching, latent_matching_score, compute_avg_FOSCTTM
from matching.models.base import ImageVAEModule, GEXADTVAEModule
from matching.models.classifier import BallsClassifier, GEXADT_Classifier
from matching.data_utils.datamodules import NoisyBallsDataModule, GEXADTDataModule, BallsDataModule
import argparse
import torch
import pandas as pd
from timeit import default_timer as timer
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Matching on balls or ADT/GEX")
    parser.add_argument("--checkpoint", metavar = "CHECKPOINT PATH", type = str)
    parser.add_argument("--dataset", metavar = "BALLS OR GEXADT", type = str, default = "GEXADT")
    parser.add_argument("--model", metavar = "VAE or CLASSIFIER", type = str, default = "CLASSIFIER")
    parser.add_argument("--gpu", action = "store_true")
    return parser.parse_args()

def load_full_data(dataset):
    """should be a torch dataset"""
    loader = torch.utils.data.DataLoader(dataset, batch_size = len(dataset))
    return next(iter(loader))

def compute_metrics(match1, match2, y, matching, data, z = None):
    traces = []
    foscttms = []
    outputs = {}
    if isinstance(data, BallsDataModule): mses, random_mses = [], []
    for label in np.unique(y):
        subset = y == label 
        x1_, x2_ = match1[subset], match2[subset]
        x1_, x2_ = x1_.reshape((x1_.shape[0], -1)).cpu().detach().numpy(), x2_.reshape((x2_.shape[0], -1)).cpu().detach().numpy()
        print(f"starting matching on label {label} with {len(x1_)} samples...")
        start = timer()
        coupling = matching(x1_, x2_)
        if isinstance(coupling, torch.Tensor): coupling = coupling.cpu().detach().numpy()
        end = timer()
        print(f"{matching} took {end - start} seconds on label {label} with {len(x1_)} samples")
        if np.isnan(coupling).any(): continue ## skip everything if any nans 

        trace = np.trace(coupling)/len(x1_)
        traces.append(trace)
        outputs[f"Trace {label}"] = trace
        x2_matched = coupling @ x2_

        if isinstance(data, GEXADTDataModule):
            FOSCTTM = compute_avg_FOSCTTM(x2_, x2_matched)
            foscttms.append(FOSCTTM)
            outputs[f"FOSCTTM {label}"] = float(FOSCTTM)
        if isinstance(data, BallsDataModule):
            z_subset = z[subset]
            z_subset = z_subset.cpu().detach().numpy()
            mse = latent_matching_score(coupling, z_subset)
            random_mse = float(((np.random.permutation(z_subset) - z_subset)**2).mean())
            mses.append(mse)
            random_mses.append(random_mse)
            outputs[f"MSE {label}"] = float(mse)
            outputs[f"Random MSE {label}"] = float(random_mse)
    
    print(outputs)

    return outputs

if __name__ == "__main__":

    args = parse_arguments()

    print(args)

    device = "cuda" if args.gpu else "cpu"

    if args.dataset == "BALLS":
        data = NoisyBallsDataModule()
        print("loading model...")
        if args.model == "CLASSIFIER":
            model = BallsClassifier.load_from_checkpoint(args.checkpoint, map_location=torch.device(device))
        if args.model == "VAE":
            model = ImageVAEModule.load_from_checkpoint(args.checkpoint, map_location=torch.device(device))
    if args.dataset == "GEXADT":
        data = GEXADTDataModule()
        print("loading model...")
        if args.model == "CLASSIFIER":
            model = GEXADT_Classifier.load_from_checkpoint(args.checkpoint, map_location=torch.device(device))
        if args.model == "VAE":
            model = GEXADTVAEModule.load_from_checkpoint(args.checkpoint, map_location=torch.device(device))
    
    model.eval()
    print("data setup...")
    data.prepare_data()
    data.setup(stage = "test")
    train_data, val_data, test_data = load_full_data(data.train_dataset), load_full_data(data.val_dataset), load_full_data(data.test_dataset)
    print("forward pass...")
    match1, match2 =  model(test_data[0].to(device), test_data[1].to(device))
    match1, match2 = match1.cpu(), match2.cpu()
    y = test_data[2]
    print("starting evaluation...")
    if isinstance(data, BallsDataModule): 
        z = test_data[3]
        outputs_EOT = compute_metrics(match1 = match1, match2 = match2, y = y, z = z, data = data, matching = eot_matching)
        outputs_kNN = compute_metrics(match1 = match1, match2 = match2, y = y, z = z, data = data, matching = snn_matching)
    if isinstance(data, GEXADTDataModule):
        outputs_EOT = compute_metrics(match1 = match1, match2 = match2, y = y, data = data, matching = eot_matching)
        outputs_kNN = compute_metrics(match1 = match1, match2 = match2, y = y, data = data, matching = snn_matching)

    outpath = "results/" + args.model + "_" + (args.checkpoint.split("/"))[-1] + args.dataset 

    
    pd.DataFrame.from_dict(data = outputs_kNN, orient = "index").to_csv(outpath + "_kNN.csv", header = False)
    pd.DataFrame.from_dict(data = outputs_EOT, orient = "index").to_csv(outpath + "_EOT.csv", header = False)

### setup with stage test
