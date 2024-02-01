import sys
import os
import numpy as np
from matching.utils import nullable_string, load_from_checkpoint_
from matching.data_utils.datamodules import  GEXADTDataModule, BallsDataModule
from matching.models.probe import MatchingProbe
import argparse
import torch
import pandas as pd
from pytorch_lightning import loggers, Trainer


def parse_arguments():
    parser = argparse.ArgumentParser(description = "Probing Experiment")
    parser.add_argument("--checkpoint", metavar = "Checkpoint, or Random, or GT", type = nullable_string)
    parser.add_argument("--max_epochs", metavar = "MAX_EPOCHS", type = int, default = 2)
    parser.add_argument("--batch_size", metavar = "BATCH_SIZE", type = int, default = 500)
    parser.add_argument("--lr", metavar = "LEARNING_RATE", type = float, default = 0.0001)
    parser.add_argument("--unbiased", action='store_true', default=False)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    print(args)

    data = GEXADTDataModule(batch_size = args.batch_size)
    if str.lower(args.checkpoint) in ["random", "gt"]:
        embedding = args.checkpoint
    else:
        embedding = load_from_checkpoint_(args.checkpoint, "GEXADT")

    probe = MatchingProbe(embedding = embedding, lr = args.lr, unbiased = args.unbiased)

    trainer = Trainer(accelerator = "gpu",
                    devices = 1,
                    log_every_n_steps = 1,
                    num_sanity_val_steps=2
                    )
    
    trainer.fit(model = probe, datamodule = data)

    