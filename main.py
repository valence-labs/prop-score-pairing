from pytorch_lightning import loggers, Trainer

from matching.models.classifier import BallsClassifier
from matching.data_utils.datamodules import BallsDataModule
from matching.callbacks import MatchingMetrics

if __name__ == "__main__":

    model = BallsClassifier(latent_dim = 64)
    data = BallsDataModule(batch_size = 100)

    logdir = "checkpoints/" + "balls/"

    trainer = Trainer(accelerator = "gpu", 
                    max_epochs = 10,
                    default_root_dir = logdir,
                    devices = 1,
                    log_every_n_steps = 1,
                    num_sanity_val_steps=0,
                    callbacks = [MatchingMetrics(run_scot=True)])
    

    trainer.fit(model = model, datamodule = data)