import sys
import os
import numpy as np
from matching.utils import scot_matching, snn_matching, eot_matching, latent_matching_score, compute_avg_FOSCTTM
from matching.models.base import ImageVAEModule, GEXADTVAEModule
from matching.models.classifier import BallsClassifier, GEXADT_Classifier
from matching.data_utils.datamodules import NoisyBallsDataModule, GEXADTDataModule, BallsDataModule
from matching.models.probe import MatchingProbe
import argparse
import torch
import pandas as pd
from pytorch_lightning import loggers, Trainer


def parse_arguments():
    parser = argparse.ArgumentParser(description = "Matching on balls or ADT/GEX")
    parser.add_argument("--max_epochs", metavar = "MAX_EPOCHS", type = int, default = 50)
    parser.add_argument("--batch_size", metavar = "BATCH_SIZE", type = int, default = 500)
    parser.add_argument("--lr", metavar = "LEARNING_RATE", type = float, default = 0.0001)
    parser.add_argument('--unbiased', action='store_true', default=False)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    print(args)

    logdir = "checkpoints/" + "probe/"


    wandb_logger = loggers.WandbLogger(save_dir = logdir, project = "Matching-Experiments")
    wandb_logger.experiment.config.update(vars(args))
    wandb_logger.experiment.config["Task"] = "PROBING_PS"


    data = GEXADTDataModule()
    embedding = GEXADT_Classifier.load_from_checkpoint("results/checkpoints/PS-epoch=06-full_val_loss=0.63.ckpt", map_location=torch.device("cuda"))
    probe = MatchingProbe(embedding = embedding, lr = args.lr, unbiased = args.unbiased)

    trainer = Trainer(accelerator = "gpu", 
                    max_epochs = args.max_epochs,
                    default_root_dir = logdir,
                    devices = 1,
                    log_every_n_steps = 1,
                    num_sanity_val_steps=2,
                    logger = wandb_logger,
                    )
    
    trainer.fit(model = probe, datamodule = data)

    wandb_logger.experiment.finish()

    wandb_logger_VAE = loggers.WandbLogger(save_dir = logdir, project = "Matching-Experiments")
    wandb_logger_VAE.experiment.config.update(vars(args))
    wandb_logger_VAE.experiment.config["Task"] = "PROBING_VAE"


    data = GEXADTDataModule()
    embedding_VAE = GEXADTVAEModule.load_from_checkpoint("results/checkpoints/VAE-epoch=239-full_val_loss=0.47.ckpt", map_location=torch.device("cuda"))
    probe_VAE = MatchingProbe(embedding = embedding_VAE, lr = args.lr, unbiased = args.unbiased)

    trainer_VAE = Trainer(accelerator = "gpu", 
                    max_epochs = args.max_epochs,
                    default_root_dir = logdir,
                    devices = 1,
                    log_every_n_steps = 1,
                    num_sanity_val_steps=2,
                    logger = wandb_logger_VAE,
                    )
    
    trainer_VAE.fit(model = probe_VAE, datamodule = data)

    wandb_logger_VAE.experiment.finish()


    wandb_logger_nomatch = loggers.WandbLogger(save_dir = logdir, project = "Matching-Experiments")
    wandb_logger_nomatch.experiment.config.update(vars(args))
    wandb_logger_nomatch.experiment.config["Task"] = "PROBING_NOMATCH"

    data_nomatch = GEXADTDataModule()
    probe_nomatch = MatchingProbe(lr = args.lr, match = "None", unbiased = args.unbiased)

    trainer_nomatch = Trainer(accelerator = "gpu", 
                    max_epochs = args.max_epochs,
                    default_root_dir = logdir,
                    devices = 1,
                    log_every_n_steps = 1,
                    num_sanity_val_steps=2,
                    logger = wandb_logger_nomatch,
                    )
    
    trainer_nomatch.fit(model = probe_nomatch, datamodule = data_nomatch)

    wandb_logger_nomatch.experiment.finish()


    wandb_logger_random = loggers.WandbLogger(save_dir = logdir, project = "Matching-Experiments")
    wandb_logger_random.experiment.config.update(vars(args))
    wandb_logger_random.experiment.config["Task"] = "PROBING_RANDOM"

    data_random = GEXADTDataModule()
    probe_random = MatchingProbe(lr = args.lr, match = "Random", unbiased = args.unbiased)

    trainer_random = Trainer(accelerator = "gpu", 
                    max_epochs = args.max_epochs,
                    default_root_dir = logdir,
                    devices = 1,
                    log_every_n_steps = 1,
                    num_sanity_val_steps=2,
                    logger = wandb_logger_random,
                    )
    
    trainer_random.fit(model = probe_random, datamodule = data_random)