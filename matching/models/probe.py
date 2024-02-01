import torch
from pytorch_lightning import LightningModule
import numpy as np
from torch import optim, nn
from torch.nn import functional as F
from ..utils import eot_matching, snn_matching
from ..data_utils.datamodules import GEXADTDataset, GEXADTDataset_Double, CoupledDataset, WrapperDataset

from typing import Union

class MSEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prediction, target, samples=None):
        if samples is None:
            samples = prediction
        delta = samples - target
        output = torch.pow(delta, 2).mean()
        for d in delta.shape:
            delta /= d
        ctx.save_for_backward(delta)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        delta, = ctx.saved_tensors
        grad_prediction = grad_target = grad_samples = None
        if ctx.needs_input_grad[0]:
            grad_prediction = grad_output * 2 * delta
        if ctx.needs_input_grad[1]:
            grad_target = -grad_output * 2 * delta

        return grad_prediction, grad_target, grad_samples
    
class IV_MSELoss(torch.nn.Module):
    """
    2SLS Unbiased Loss
    """
    def __init__(self):
        super(IV_MSELoss, self).__init__()

    def forward(self, prediction, target, samples=None):
        if prediction.shape[0] != target.shape[0]:
            raise RuntimeError(
                "Size mismatch between prediction (%s) and target (%s)"
                % (prediction.shape, target.shape)
            )
        if samples is not None and prediction.shape[0] != samples.shape[0]:
            raise RuntimeError(
                "Size mismatch between prediction (%s) and samples (%s)"
                % (prediction.shape, samples.shape)
            )

        return MSEFunction.apply(prediction, target, samples)

class MatchingProbe(LightningModule):
    def __init__(self,
                 embedding: Union[LightningModule, str] = None, ## pre-trained embedding, or string "random" or "gt" for random matching or ground truth matching
                 lr: float = 0.001, 
                 wd: float = 0.0005,
                 unbiased: bool = False
                 ):
        super().__init__()

        self.lr = lr
        self.wd = wd
        self.embedding = embedding
        self.unbiased = unbiased
        
        if self.unbiased: 
            self.IVloss = IV_MSELoss()
        self.MSEloss = torch.nn.MSELoss()
        if isinstance(self.embedding, LightningModule):
            self.embedding.freeze()
        else:
            self.embedding = str.lower(self.embedding)

    def forward(self, x):
        pred = self.probe(x)
        return pred
    
    def step(self, batch, batch_idx, unbiased = False):
        if unbiased:
            x1, x1_double, x2, y = batch
            pred = self.probe(x1)
            with torch.no_grad():
                sample = self.probe(x1_double)
            loss = self.IVloss(pred, x2, sample)
        else:
            x1, x2, y = batch
            pred = self.probe(x1)
            loss = self.MSEloss(pred, x2)
        return loss
    
    def training_step(self, batch, batch_idx):
        ## dataloaders should yield modalities x1, x2, (x2_double), and label y 
        loss = self.step(batch, batch_idx, self.unbiased)
        self.log("Train Loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        ## we only use validation to see how the classifier is doing 
        ## switch to x2 predicting x1
        loss = self.step(batch, batch_idx, unbiased = False)
        x1, x2, y = batch
        baseline_pred = x2.mean(dim = 0, keepdim=True).expand(x2.shape[0], -1)
        baseline_loss = self.MSEloss(baseline_pred, x2)
        r2 = 1 - loss/baseline_loss
        self.log("Val Baseline Loss", baseline_loss, on_epoch = True)
        self.log("Val Loss", loss, on_epoch = True)
        self.log("Val R2", r2, on_epoch = True)

        # if self.unbiased == True:
        #     with torch.no_grad():
        #         if self.embedding == "random":
        #             coupling = torch.full((x2.shape[0], x2.shape[0]), torch.tensor(1/x2.shape[0]), device = "cuda")
        #         elif isinstance(self.embedding, LightningModule):
        #             match1, match2 = self.embedding(x1, x2)
        #             coupling = eot_matching(match1, match2)

        #         pred = self.probe(x1)
        #         pred_projected = torch.t(coupling) @ pred
        #         loss_projected = self.loss(pred_projected, x2)

        #         self.log("Val Loss Projected", loss_projected)

    def on_validation_epoch_end(self): 
        pass

    def test_step(self, batch, batch_idx):
        ## we only use validation to see how the classifier is doing 
        loss = self.step(batch, batch_idx)
        x1, x2, y = batch
        baseline_pred = x2.mean(dim = 0, keepdim=True).expand(x2.shape[0], -1)
        baseline_loss = self.MSEloss(baseline_pred, x2)
        r2 = 1 - loss/baseline_loss
        self.log("Test Baseline Loss", baseline_loss, on_epoch = True)
        self.log("Test Loss", loss, on_epoch = True)
        self.log("Test R2", r2, on_epoch = True)

        # if self.unbiased == True:
        #     with torch.no_grad():
        #         if self.embedding == "random":
        #             coupling = torch.full((x2.shape[0], x2.shape[0]), torch.tensor(1/x2.shape[0]), device = "cuda")
        #         elif isinstance(self.embedding, LightningModule):
        #             match1, match2 = self.embedding(x1, x2)
        #             coupling = eot_matching(match1, match2)

        #         pred = self.probe(x1)
        #         pred_projected = torch.t(coupling) @ pred
        #         loss_projected = self.loss(pred_projected, x2)

        #         self.log("Test Loss Projected", loss_projected)

    def on_test_epoch_end(self):
        pass

    def setup(self, stage:str):
        test_loader = self.trainer.datamodule.test_dataloader()
        batch1, batch2, _ = next(iter(test_loader))
        num_input = batch1.shape[1]
        num_output = batch2.shape[1]
        self.probe = nn.Sequential(nn.Linear(num_input, num_input * 2),
                        nn.BatchNorm1d(num_input * 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(num_input * 2, num_output))
        
        ## match the data before training to modify the dataset, or do nothing if ground truth
        
        # if self.embedding == "gt":
        #     if self.unbiased:
        #         self.trainer.datamodule.train_dataset = GEXADTDataset_Double(self.trainer.datamodule.train_data_adt, 
        #                                                               self.trainer.datamodule.train_data_adt.clone(),
        #                                                               self.trainer.datamodule.train_data_gex,
        #                                                               self.trainer.datamodule.train_labels)
        #     return None
        
        
        with torch.no_grad():
            x1 = self.trainer.datamodule.train_data_adt.to("cuda")
            x2 = self.trainer.datamodule.train_data_gex.to("cuda")
            if self.unbiased: x1_clone = torch.zeros_like(x1).to("cuda")
            y = self.trainer.datamodule.train_labels.to("cuda")

            coupled_datasets = []

            for label in torch.unique(y):
                subset = y == label
                x1_, x2_ = x1[subset].clone(), x2[subset].clone()
                if isinstance(self.embedding, LightningModule):
                    x1_ = x1_[torch.randperm(x1_.shape[0])]  ## shuffle to avoid pathologies 
                    match1, match2 = self.embedding(x1_, x2_)
                    coupling = eot_matching(match1, match2)
                elif self.embedding == "random":
                    x1_ = x1_[torch.randperm(x1_.shape[0])]  ## shuffle to avoid pathologies 
                    coupling = torch.full((x2_.shape[0], x2_.shape[0]), torch.tensor(1/x2_.shape[0]), device = "cuda") ## 1/n in coupling matrix everywhere
                elif self.embedding == "gt":
                    coupling = torch.eye(x2_.shape[0], device = "cuda") ## ground truth matching

                coupled_datasets.append(CoupledDataset(x1_, x2_, torch.t(coupling), self.unbiased))  ## implement self.unbiased inside CoupledDataset


                # idx = torch.multinomial(torch.t(coupling), num_samples = 1)  ## remove torch.t to sample from x2 instead
                # x1_clone_1 = (x1_.clone())[torch.flatten(idx)].view(x1_.size())
                # x1[subset] = x1_clone_1
                # if self.unbiased: 
                #     idx = torch.multinomial(torch.t(coupling), num_samples = 1)
                #     x1_clone_2 = (x1_.clone())[torch.flatten(idx)].view(x1_.size())
                #     x1_clone[subset] = x1_clone_2

        # if self.unbiased:
        #     self.trainer.datamodule.train_dataset = GEXADTDataset_Double(x1.to("cpu"), x1_clone.to("cpu"), x2.to("cpu"), y.to("cpu"))      
    
        self.trainer.datamodule.train_dataset = WrapperDataset(coupled_datasets)
        
    def on_train_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr = self.lr, weight_decay = self.wd) ## don't train encoder
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.lr, total_steps = self.trainer.max_epochs, pct_start = 0.1)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    