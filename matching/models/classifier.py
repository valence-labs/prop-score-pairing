import sys
import numpy as np
import torch
from torch import optim, nn

from .encoders import ImageEncoder
from .base import BaseClassifier

class BallsClassifier(BaseClassifier):
    def __init__(self, 
                latent_dim: int = 128,
                **kwargs):
        
        super().__init__(**kwargs)

        self.Encoder_1 = ImageEncoder(latent_dim)
        self.logits_1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 12)
        )
        self.Encoder_2 = ImageEncoder(latent_dim)
        self.logits_2 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 12)
        )
        self.clf1 = torch.nn.Sequential(self.Encoder_1, self.logits_1)
        self.clf2 = torch.nn.Sequential(self.Encoder_2, self.logits_2)
    
    def training_step(self, batch, batch_idx):
        x1, x2, y, z = batch
        loss_dict = super().training_step(batch = (x1, x2, y), batch_idx = batch_idx)
        return loss_dict

class GEXADT_Classifier(BaseClassifier):
    def __init__(self, 
                 n_classes: int = 45,
                 n_hidden: int = 1024,
                **kwargs):
        super().__init__(**kwargs)
        ### implement clf1 as adt classifier
        ### implement clf2 as gex classifier

        ## Encoder from Caroline's lab
                    
        self.encoder_1 = nn.Sequential(nn.LazyLinear(out_features = n_hidden),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(n_hidden),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden),
                                )
        
        self.encoder_2 = nn.Sequential(nn.LazyLinear(out_features = n_hidden),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(n_hidden),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden),
                                )
        
        self.clf1 = torch.nn.Sequential(
            self.encoder_1,
            torch.nn.Linear(n_hidden, n_classes)
        )
        self.clf2 = torch.nn.Sequential(
            self.encoder_2,
            torch.nn.Linear(n_hidden, n_classes)
        )

