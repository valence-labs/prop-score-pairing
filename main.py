from pytorch_lightning import loggers, Trainer
import argparse
from matching.models.classifier import BallsClassifier, GEXADT_Classifier
from matching.data_utils.datamodules import NoisyBallsDataModule, GEXADTDataModule
from matching.callbacks import MatchingMetrics
from pytorch_lightning.callbacks import ModelCheckpoint

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Matching on balls or ADT/GEX")
    parser.add_argument("--max_epochs", metavar = "MAX_EPOCHS", type = int, default = 50)
    parser.add_argument("--batch_size", metavar = "BATCH_SIZE", type = int, default = 256)
    parser.add_argument("--lr", metavar = "LEARNING_RATE", type = float, default = 0.0001)
    parser.add_argument("--dataset", metavar = "BALLS OR GEXADT", type = str, default = "GEXADT")
    parser.add_argument("--eval_interval", metavar = "INTERVAL OF MATCHING METRICS", type = int, default = 10)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    logdir = "checkpoints/" + "balls/"
    wandb_logger = loggers.WandbLogger(save_dir = logdir, project = "Matching-Experiments")
    wandb_logger.experiment.config.update(vars(args))
    wandb_logger.experiment.config["model"] = "PS"

    checkpoint_callback = ModelCheckpoint(
        dirpath = "results/checkpoints/",
        filename = "PS-{epoch:02d}-{full_val_loss:.2f}",
        monitor = "full_val_loss"
    )

    if args.dataset == "BALLS":
        model = BallsClassifier(latent_dim = 128, lr = args.lr)
        data = NoisyBallsDataModule(batch_size = args.batch_size)
    if args.dataset == "GEXADT":
        model = GEXADT_Classifier(n_classes = 45, lr = args.lr)
        data = GEXADTDataModule(batch_size = args.batch_size)


    trainer = Trainer(accelerator = "gpu", 
                    max_epochs = args.max_epochs,
                    default_root_dir = logdir,
                    devices = 1,
                    log_every_n_steps = 1,
                    num_sanity_val_steps=2,
                    callbacks = [MatchingMetrics(run_scot=False, eval_interval = args.eval_interval), checkpoint_callback], 
                    logger = wandb_logger,
                    )
    

    trainer.fit(model = model, datamodule = data)