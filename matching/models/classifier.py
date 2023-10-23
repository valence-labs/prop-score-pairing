import sys
import numpy as np
import torch
from torch import optim

from .encoders import ImageEncoder
from .base import BaseClassifier

class BallsClassifier(BaseClassifier):
    def __init__(self, 
                latent_dim: int = 32,
                rnlr: float = 0.0001,
                rnmomentum: float = 0.9,
                rnwd: float = 0.0001,
                **kwargs):
        
        super().__init__(**kwargs)

        self.rnlr = rnlr
        self.rnmomentum = rnmomentum
        self.rnwd = rnwd

        self.Encoder_1 = ImageEncoder(latent_dim)
        self.logits_1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 12)
        )
        self.Encoder_2 = ImageEncoder(latent_dim)
        self.logits_2 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 12)
        )
        self.clf1 = torch.nn.Sequential(self.Encoder_1, self.logits_1)
        self.clf2 = torch.nn.Sequential(self.Encoder_1, self.logits_1)

    def configure_optimizers(self):
        optimizer = optim.SGD([{'params':self.Encoder_1.base_model.parameters(), 'lr':self.rnlr, 'momentum': self.rnmomentum, 'weight_decay':self.rnwd}, 
                          {'params':self.Encoder_2.base_model.parameters(), 'lr':self.rnlr, 'momentum':self.rnmomentum, 'weight_decay':self.rnwd},
                          {'params':self.Encoder_1.fc_net.parameters()},
                          {'params':self.Encoder_2.fc_net.parameters()},
                          {'params':self.logits_1.parameters()},
                          {'params':self.logits_2.parameters()}], 
                          lr = self.lr, weight_decay = self.wd, momentum = self.momentum)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = [self.rnlr, self.rnlr, self.lr, self.lr, self.lr, self.lr], total_steps = self.trainer.max_epochs, pct_start = 0.1)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    
    def training_step(self, batch, batch_idx):
        x1, x2, y, z = batch
        loss_dict = super().training_step(batch = (x1, x2, y), batch_idx = batch_idx)
        return loss_dict

class GEXADT_Classifier(BaseClassifier):
    def __init__(self, 
                 n_classes: int = 45,
                **kwargs):
        super().__init__(**kwargs)
        ### implement clf1 as adt classifier
        ### implement clf2 as gex classifier
        self.clf1 = torch.nn.Sequential(
            torch.nn.LazyLinear(out_features=256),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(out_features=256),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(out_features=n_classes)
        )
        self.clf2 = torch.nn.Sequential(
            torch.nn.LazyLinear(out_features=256),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(out_features=256),
            torch.nn.ReLU(),
            torch.nn.LazyLinear(out_features=n_classes)
        )

