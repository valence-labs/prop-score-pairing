from matching.utils import snn_matching, eot_matching
from matching.callbacks import compute_metrics
from matching.models.base import ImageVAEModule, GEXADTVAEModule
from matching.models.classifier import BallsClassifier, GEXADT_Classifier
from matching.data_utils.datamodules import NoisyBallsDataModule, GEXADTDataModule, BallsDataModule
import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd

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
    test_data = DataLoader(data.test_dataset, batch_size = 256, shuffle = False)
    print("forward pass...")
    with torch.no_grad():
        for (i, batch) in enumerate(test_data):
            if i == 0:
                match1, match2 =  model(batch[0].to(device), batch[1].to(device))
                y = batch[2]
                if isinstance(data, BallsDataModule): 
                    z = batch[3]
            else:
                match1_, match2_ = model(batch[0].to(device), batch[1].to(device))
                match1 = torch.cat((match1, match1_), 0)
                match2 = torch.cat((match2, match2_), 0)
                y = torch.cat((y, batch[2]), 0)
                if isinstance(data, BallsDataModule): 
                    z = torch.cat((z, batch[3]), 0)
    match1, match2 = match1.cpu(), match2.cpu()
    print("starting evaluation...")
    if isinstance(data, BallsDataModule): 
        outputs_EOT = compute_metrics(match1 = match1, match2 = match2, y = y, z = z, data = data, matching = eot_matching)
        outputs_kNN = compute_metrics(match1 = match1, match2 = match2, y = y, z = z, data = data, matching = snn_matching)
    if isinstance(data, GEXADTDataModule):
        outputs_EOT = compute_metrics(match1 = match1, match2 = match2, y = y, data = data, matching = eot_matching)
        outputs_kNN = compute_metrics(match1 = match1, match2 = match2, y = y, data = data, matching = snn_matching)

    outpath = "results/" + args.model + "_" + (args.checkpoint.split("/"))[-1] + args.dataset 

    pd.DataFrame.from_dict(data = outputs_kNN, orient = "index").to_csv(outpath + "_kNN.csv", header = False)
    pd.DataFrame.from_dict(data = outputs_EOT, orient = "index").to_csv(outpath + "_EOT.csv", header = False)

