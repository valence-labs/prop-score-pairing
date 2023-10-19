import torch
from pytorch_lightning import LightningModule
import numpy as np
from torch import optim
from .. import compute_class_weights, convert_to_labels

class BaseClassifier(LightningModule):
    def __init__(self, 
                 lr: float = 0.01, 
                 wd: float = 0.0001,
                 momentum: float = 0.0):
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
        for label in np.unique(y.cpu().detach().numpy()):
            self.train_match_x1[label] = np.concatenate((self.train_match_x1[label], logits_1[y == label].cpu().detach().numpy()))
            self.train_match_x2[label] = np.concatenate((self.train_match_x1[label], logits_2[y == label].cpu().detach().numpy()))
        return {"loss": loss_1 + loss_2,
                "md1_loss": loss_1,
                "md2_loss": loss_2}
    
    def validation_step(self, batch, batch_idx):
        ## we only use validation to see how the classifier is doing 
        loss_dict = self.training_step(batch, batch_idx)
        self.log("val loss 1", loss_dict["md1_loss"])
        self.log("val loss 2", loss_dict["md2_loss"])

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
        labels = convert_to_labels(self.trainer.datamodule.y_tr) # type: ignore[attr-defined]
        num_classes, class_weights = compute_class_weights(labels)
        self.train_match_x1 = {label:np.empty((0, num_classes)) for label in np.unique(labels)}
        self.train_match_x2 = {label:np.empty((0, num_classes)) for label in np.unique(labels)}
        self.loss = torch.nn.CrossEntropyLoss(weight = torch.from_numpy(class_weights).float())

    def on_train_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr = self.lr, weight_decay = self.wd, momentum = self.momentum)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.lr, total_steps = self.trainer.max_epochs, pct_start = 0.1)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    