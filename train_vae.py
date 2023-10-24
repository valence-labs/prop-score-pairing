from pytorch_lightning import loggers, Trainer
import argparse
from matching.models.base import ImageVAEModule, GEXADTVAEModule
from matching.data_utils.datamodules import NoisyBallsDataModule, GEXADTDataModule
from matching.callbacks import MatchingMetrics

if __name__ == "__main__":

    logdir = "checkpoints/" + "balls/" + "vae/"
    wandb_logger = loggers.WandbLogger(save_dir = logdir)

    model = ImageVAEModule()
    data = NoisyBallsDataModule(batch_size = 100)

    trainer = Trainer(accelerator = "gpu", 
                    max_epochs = 50,
                    default_root_dir = logdir,
                    devices = 1,
                    log_every_n_steps = 1,
                    num_sanity_val_steps=2,
                    callbacks = [MatchingMetrics(run_scot=False, eval_interval = 10)], 
                    logger = wandb_logger
                    )
    
    trainer.fit(model = model, datamodule = data)