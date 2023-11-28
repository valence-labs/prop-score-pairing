from pytorch_lightning import loggers, Trainer
import argparse
from matching.models.classifier import BallsClassifier, GEXADT_Classifier
from matching.models.vae import ImageVAEModule, GEXADTVAEModule
from matching.data_utils.datamodules import NoisyBallsDataModule, GEXADTDataModule
from matching.callbacks import MatchingMetrics
from pytorch_lightning.callbacks import ModelCheckpoint

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Matching on balls or ADT/GEX")
    parser.add_argument("--max_epochs", metavar = "MAX_EPOCHS", type = int, default = 50)
    parser.add_argument("--batch_size", metavar = "BATCH_SIZE", type = int, default = 256)
    parser.add_argument("--lr", metavar = "LEARNING_RATE", type = float, default = 0.0001)
    parser.add_argument("--dataset", metavar = "BALLS OR GEXADT", type = str, default = "GEXADT")
    parser.add_argument("--eval_interval", metavar = "INTERVAL OF MATCHING METRICS", type = int, default = 1)
    parser.add_argument("--run_scot", action = "store_true")
    parser.add_argument("--model", metavar = "PS OR VAE", type = str, default = "PS")
    parser.add_argument("--lamb", metavar = "WEIGHT ON KL TERM", type = float, default = 0.0000000001)
    parser.add_argument("--alpha", metavar = "WEIGHT ON MODALITY CLASSIFIER", type = float, default = 1)
    parser.add_argument("--beta", metavar = "WEIGHT ON LABEL CLASSIFIER", type = float, default = 0.1)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    assert args.model in ["PS", "VAE"]

    logdir = "checkpoints/" + "balls/" + args.model + "/"
    wandb_logger = loggers.WandbLogger(save_dir = logdir, project = "Matching-Experiments")
    wandb_logger.experiment.config.update(vars(args))
    wandb_logger.experiment.config["model"] = args.model

    checkpoint_callback = ModelCheckpoint(
        dirpath = "results/checkpoints/",
        filename = args.model + "-{epoch:02d}-{full_val_loss:.2f}",
        monitor = "full_val_loss"
    )

    if args.dataset == "BALLS":
        if args.model == "PS":
            model = BallsClassifier(latent_dim = 128, lr = args.lr)
        elif args.model == "VAE":
            model = ImageVAEModule(alpha = args.alpha, beta = args.beta, lamb = args.lamb, lr = args.lr)
        data = NoisyBallsDataModule(batch_size = args.batch_size)
    if args.dataset == "GEXADT":
        if args.model == "PS":
            model = GEXADT_Classifier(n_classes = 45, lr = args.lr)
        elif args.model == "VAE":
            model = GEXADTVAEModule(alpha = args.alpha, beta = args.beta, lamb = args.lamb, lr = args.lr)
        data = GEXADTDataModule(batch_size = args.batch_size)


    trainer = Trainer(accelerator = "gpu", 
                    max_epochs = args.max_epochs,
                    default_root_dir = logdir,
                    devices = 1,
                    log_every_n_steps = 1,
                    num_sanity_val_steps=2,
                    callbacks = [MatchingMetrics(run_scot=args.run_scot, eval_interval = args.eval_interval), checkpoint_callback], 
                    logger = wandb_logger,
                    )
    
    trainer.fit(model = model, datamodule = data)