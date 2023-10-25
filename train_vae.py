from pytorch_lightning import loggers, Trainer
import argparse
from matching.models.base import ImageVAEModule, GEXADTVAEModule
from matching.data_utils.datamodules import NoisyBallsDataModule, GEXADTDataModule
from matching.callbacks import MatchingMetrics

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Matching on balls or ADT/GEX")
    parser.add_argument("--max_epochs", metavar = "MAX_EPOCHS", type = int, default = 50)
    parser.add_argument("--batch_size", metavar = "BATCH_SIZE", type = int, default = 256)
    parser.add_argument("--lr", metavar = "LEARNING_RATE", type = float, default = 0.0005)
    parser.add_argument("--dataset", metavar = "BALLS OR GEXADT", type = str, default = "GEXADT")
    parser.add_argument("--eval_interval", metavar = "INTERVAL OF MATCHING METRICS", type = int, default = 10)
    parser.add_argument("--lamb", metavar = "WEIGHT ON KL TERM", type = float, default = 0.0000000001)
    parser.add_argument("--alpha", metavar = "WEIGHT ON MODALITY CLASSIFIER", type = float, default = 1)
    parser.add_argument("--beta", metavar = "WEIGHT ON LABEL CLASSIFIER", type = float, default = 0.1)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    print(args)

    logdir = "checkpoints/" + "balls/" + "vae/"
    wandb_logger = loggers.WandbLogger(save_dir = logdir)
    wandb_logger.experiment.config.update(vars(args))
    wandb_logger.experiment.config["model"] = "VAE"

    if args.dataset == "BALLS":
        model = ImageVAEModule(alpha = args.alpha, beta = args.beta, lamb = args.lamb, lr = args.lr)
        data = NoisyBallsDataModule(batch_size = args.batch_size)
    if args.dataset == "GEXADT":
        model = GEXADTVAEModule(lamb = args.lamb, lr = args.lr)
        data = GEXADTDataModule(batch_size = args.batch_size)

    

    trainer = Trainer(accelerator = "gpu", 
                    max_epochs = args.max_epochs,
                    default_root_dir = logdir,
                    devices = 1,
                    log_every_n_steps = 1,
                    num_sanity_val_steps=2,
                    callbacks = [MatchingMetrics(run_scot=False, eval_interval = args.eval_interval)], 
                    logger = wandb_logger
                    )
    
    trainer.fit(model = model, datamodule = data)