from matching.utils import snn_matching, eot_matching, scot_matching, load_from_checkpoint_
from matching.callbacks import compute_metrics
from matching.models.vae import ImageVAEModule, GEXADTVAEModule
from matching.models.classifier import BallsClassifier, GEXADT_Classifier
from matching.data_utils.datamodules import NoisyBallsDataModule, GEXADTDataModule, BallsDataModule
import argparse
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Matching on balls or ADT/GEX")
    parser.add_argument("--checkpoint", metavar = "CHECKPOINT PATH", type = str)
    parser.add_argument("--dataset", metavar = "BALLS OR GEXADT", type = str, default = "GEXADT")
    parser.add_argument("--run_scot", action = "store_true")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "BALLS":
        data = NoisyBallsDataModule()
    if args.dataset == "GEXADT":
        data = GEXADTDataModule()
    
    print("loading model...")
    model = load_from_checkpoint_(args.checkpoint, args.dataset, device)
    model.eval()
    print("data setup...")
    data.prepare_data()
    data.setup(stage = "test")
    test_data = DataLoader(data.test_dataset, batch_size = 256, shuffle = False)
    print("forward pass...")
    with torch.no_grad():
        for (i, batch) in tqdm(enumerate(test_data)):
            if i == 0:
                x1, x2 = batch[0], batch[1]
                match1, match2 = model(batch[0].to(device), batch[1].to(device))
                y = batch[2]
                if isinstance(data, BallsDataModule): 
                    z = batch[3]
                else:
                    z = None
            else:
                match1_, match2_ = model(batch[0].to(device), batch[1].to(device))
                x1 = torch.cat((x1, batch[0]), 0)
                x2 = torch.cat((x2, batch[1]), 0)
                match1 = torch.cat((match1, match1_), 0)
                match2 = torch.cat((match2, match2_), 0)
                y = torch.cat((y, batch[2]), 0)
                if isinstance(data, BallsDataModule): 
                    z = torch.cat((z, batch[3]), 0)
    print("starting evaluation...")
    if args.run_scot: outputs_SCOT = compute_metrics(match1 = x1, match2 = x2, y = y, z = z, data = data, matching = scot_matching)
    outputs_EOT = compute_metrics(match1 = match1, match2 = match2, y = y, z = z, data = data, matching = eot_matching)
    outputs_kNN = compute_metrics(match1 = match1, match2 = match2, y = y, z = z, data = data, matching = snn_matching)

    outpath = "results/" + str(model.__class__.__name__) + "_" + (args.checkpoint.split("/"))[-1] + args.dataset 

    pd.DataFrame.from_dict(data = outputs_kNN, orient = "index").to_csv(outpath + "_kNN.csv", header = False)
    pd.DataFrame.from_dict(data = outputs_EOT, orient = "index").to_csv(outpath + "_EOT.csv", header = False)
    if args.run_scot: pd.DataFrame.from_dict(data = outputs_SCOT, orient = "index").to_csv(outpath + "_SCOT.csv", header = False)

    print(f"Results saved under {os.getcwd() + '/' + outpath}.")
