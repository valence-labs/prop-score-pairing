import torch
from pytorch_lightning import LightningModule
import numpy as np
from torch import optim, nn
from torch.nn import functional as F

class BaseClassifier(LightningModule):
    def __init__(self, 
                 lr: float = 0.0005, 
                 wd: float = 0.0001,
                 momentum: float = 0.9):
        super().__init__()

        self.lr = lr
        self.wd = wd
        self.momentum = momentum
        self.loss = torch.nn.CrossEntropyLoss()

        ## subclasses should define self.clf1, self.clf2 classifiers for each modality in their init

    def forward(self, x1, x2):
        ## x1, x2 are different views of the scene
        logits_1 = self.clf1(x1)
        logits_2 = self.clf2(x2)
        return logits_1, logits_2
    
    def training_step(self, batch, batch_idx):
        ## dataloaders should yield modalities x1, x2 and label y 
        x1, x2, y = batch
        logits_1, logits_2 = self(x1, x2)
        loss_1 = self.loss(logits_1, y)
        loss_2 = self.loss(logits_2, y)
        self.log("train loss 1", loss_1)
        self.log("train loss 2", loss_2)
        return {"loss": loss_1 + loss_2,
                "md1_loss": loss_1,
                "md2_loss": loss_2}
    
    def validation_step(self, batch, batch_idx):
        ## we only use validation to see how the classifier is doing 
        loss_dict = self.training_step(batch, batch_idx)
        self.log("val loss 1", loss_dict["md1_loss"])
        self.log("val loss 2", loss_dict["md2_loss"])
        self.log("full_val_loss", 0.5*loss_dict["loss"]) ## used for checkpointing

    def on_validation_epoch_end(self): 
        pass

    def test_step(self, batch, batch_idx):
        ## we only use test to see how the classifier is doing 
        loss_dict = self.training_step(batch, batch_idx)
        self.log("test loss 1", loss_dict["md1_loss"])
        self.log("test loss 2", loss_dict["md2_loss"])

    def on_test_epoch_end(self):
        pass

    def setup(self, stage:str):
        ## placeholders to compute matching metrics on the fly
        pass

    def on_train_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.wd)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.lr, total_steps = self.trainer.max_epochs, pct_start = 0.1)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    
class BaseVAEModule(LightningModule):
    def __init__(self, 
                 lr: float = 0.0001, 
                 wd: float = 0.00001,
                 momentum: float = 0.8,
                 lamb: float = 0.0000000001,
                 alpha: float = 1,
                 beta: float = 0,
                 num_classes: int = 45,
                 latent_dim: int = 128):
        super().__init__()

        self.lr = lr
        self.wd = wd
        self.momentum = momentum
        self.lamb = lamb
        self.alpha = alpha
        self.beta = beta

        ## subclasses should define self.model1, model2 VAEs for each modality in their init
        ## vaes should have attribute self.model1.latent_dim

        self.CE_Cond = torch.nn.CrossEntropyLoss()  # torch.nn.CrossEntropyLoss(weight = torch.from_numpy(class_weights).float())
        self.CE = torch.nn.CrossEntropyLoss()

        ## taken from caroline's paper
        
        self.condclf = nn.Sequential(
            nn.Linear(latent_dim, num_classes),
        )

        self.clf = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,2) ## 2 is number of modalities
        )


    def forward(self, x1, x2):
        ## x1, x2 are different views of the scene
        ## primarily used for forward call in callbacks
        latent_1 = self.model1(x1)[1]  ## pick out latent 
        latent_2 = self.model2(x2)[1] 
        return latent_1, latent_2
    
    def training_step(self, batch, batch_idx):
        ## dataloaders should yield modalities x1, x2 and label y 
        x1 = batch[0]
        x2 = batch[1]
        y = batch[2]
        VAE_1_output = self.model1(x1)
        VAE_2_output = self.model2(x2)
        ### unpack
        recon_1, z_1, mu_1, log_var_1 = VAE_1_output
        recon_2, z_2, mu_2, log_var_2 = VAE_2_output
        loss_1 = self.loss_function(recon_1, x1, mu_1, log_var_1)
        loss_2 = self.loss_function(recon_2, x2, mu_2, log_var_2)
        
        x1_labels = torch.zeros(x1.size(0),).long().to("cuda")
        x2_labels = torch.ones(x2.size(0),).long().to("cuda")

        x1_scores_clf = self.clf(z_1)
        x2_scores_clf = self.clf(z_2)

        x1_scores_condclf = self.condclf(z_1)
        x2_scores_condclf = self.condclf(z_2)

        for key in loss_1:
            self.log(f"{key} Model 1", loss_1[key])
            self.log(f"{key} Model 2", loss_2[key])
        
        loss_clf = 0.5 * (self.CE(x1_scores_clf, x1_labels) + self.CE(x2_scores_clf, x2_labels))
        loss_condclf = 0.5 * (self.CE_Cond(x1_scores_condclf, y) + self.CE_Cond(x2_scores_condclf, y))

        self.log("Modality Classifier Loss", loss_clf)
        self.log("Label Classifier Loss", loss_condclf)

        loss = loss_1["vae_loss"] + loss_2["vae_loss"] + self.alpha * loss_clf + self.beta * loss_condclf

        self.log("VAE Total Training Loss", loss)

        return {"loss": loss,
                "md1_loss": loss_1["vae_loss"],
                "md2_loss": loss_2["vae_loss"]}

    ## taken from ## adapted from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py    
    def loss_function(self, 
                      recons,
                      input,
                      mu,
                      log_var) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + self.lamb * kld_loss
        return {'vae_loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    
    def validation_step(self, batch, batch_idx):
        loss_dict = self.training_step(batch, batch_idx)
        self.log("full_val_loss", loss_dict["loss"])  ## flag for checkpointing

    def on_validation_epoch_end(self): 
        pass

    def test_step(self, batch, batch_idx):
        ## we only use test to see how the classifier is doing 
        loss_dict = self.training_step(batch, batch_idx)
        self.log("test loss 1", loss_dict["md1_loss"])
        self.log("test loss 2", loss_dict["md2_loss"])

    def on_test_epoch_end(self):
        pass

    def setup(self, stage:str):
        pass

    def on_train_epoch_end(self):
        pass

    def configure_optimizers(self):
        ## change to adam 
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay = self.wd)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.lr, total_steps = self.trainer.max_epochs, pct_start = 0.1)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    
