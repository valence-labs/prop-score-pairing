import torch
from pytorch_lightning import LightningModule
import numpy as np
from torch import optim, nn
from torch.nn import functional as F
from ..utils import eot_matching, snn_matching
from ..data_utils.datamodules import GEXADTDataset, GEXADTDataset_Double

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
                 embedding = None, ## pre-trained embedding
                 lr: float = 0.001, 
                 wd: float = 0.0005,
                 match: str = "PS", ## "PS" or "None" or "Random",
                 unbiased: bool = False
                 ):
        super().__init__()

        self.lr = lr
        self.wd = wd
        assert match in ["PS", "None", "Random"]
        self.match = match
        self.embedding = embedding
        self.unbiased = unbiased
        
        if self.unbiased: 
            self.loss = IV_MSELoss()
        else:
            self.loss = torch.nn.MSELoss()
        if embedding is not None:
            self.embedding.freeze()

    def forward(self, x):
        ## x1, x2 are different views of the scene
        pred = self.probe(x)
        return pred
    
    def _training_step(self, batch, batch_idx, unbiased = False):
        if unbiased:
            x1, x1_double, x2, y = batch
            pred = self.probe(x1)
            with torch.no_grad():
                sample = self.probe(x1_double)
            loss = self.loss(pred, x2, sample)
        else:
            x1, x2, y = batch
            pred = self.probe(x1)
            loss = self.loss(pred, x2)
        return loss
    
    def training_step(self, batch, batch_idx):
        ## dataloaders should yield modalities x1, x2, (x2_double), and label y 
        loss = self._training_step(batch, batch_idx, self.unbiased)
        self.log("Train Loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        ## we only use validation to see how the classifier is doing 
        ## switch to x2 predicting x1
        loss = self._training_step(batch, batch_idx)
        x1, x2, y = batch
        baseline_pred = x2.mean(dim = 0, keepdim=True).expand(x2.shape[0], -1)
        baseline_loss = self.loss(baseline_pred, x2)
        r2 = 1 - loss/baseline_loss
        self.log("Val Baseline Loss", baseline_loss, on_epoch = True)
        self.log("Val Loss", loss, on_epoch = True)
        self.log("Val R2", r2, on_epoch = True)

        with torch.no_grad():
            if self.embedding is None:
                coupling = torch.full((x2.shape[0], x2.shape[0]), torch.tensor(1/x2.shape[0]), device = "cuda")
            else:
                match1, match2 = self.embedding(x1, x2)
                coupling = eot_matching(match1, match2)

            
            pred = self.probe(x1)
            pred_projected = torch.t(coupling) @ pred
            loss_projected = self.loss(pred_projected, x2)

            self.log("Val Loss Projected", loss_projected)

    def on_validation_epoch_end(self): 
        pass

    def test_step(self, batch, batch_idx):
        ## we only use validation to see how the classifier is doing 
        loss = self._training_step(batch, batch_idx)
        self.log("test loss", loss)

    def on_test_epoch_end(self):
        pass

    def setup(self, stage:str):
        ## placeholders to compute matching metrics on the fly
        test_loader = self.trainer.datamodule.test_dataloader()
        batch1, batch2, _ = next(iter(test_loader))
        num_input = batch1.shape[1]
        num_output = batch2.shape[1]
        self.probe = nn.Sequential(nn.Linear(num_input, num_input * 2),
                        nn.BatchNorm1d(num_input * 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(num_input * 2, num_output))
        
        ## match the data before training
        
        if self.match == "None":
            if self.unbiased:
                self.trainer.datamodule.train_dataset = GEXADTDataset_Double(self.trainer.datamodule.train_data_adt, 
                                                                      self.trainer.datamodule.train_data_adt.clone(),
                                                                      self.trainer.datamodule.train_data_gex,
                                                                      self.trainer.datamodule.train_labels)

        if self.match in ["PS", "Random"]:
            with torch.no_grad():
                x1 = self.trainer.datamodule.train_data_adt.to("cuda")
                x2 = self.trainer.datamodule.train_data_gex.to("cuda")
                if self.unbiased: x1_clone = torch.zeros_like(x1).to("cuda")
                y = self.trainer.datamodule.train_labels.to("cuda")
                for label in torch.unique(y):
                    subset = y == label
                    x1_, x2_ = x1[subset].clone(), x2[subset].clone()
                    #x2_ = x2_[torch.randperm(x2_.shape[0])]
                    x1_ = x1_[torch.randperm(x2_.shape[0])]
                    if self.match == "PS":
                        match1, match2 = self.embedding(x1_, x2_)
                        coupling = eot_matching(match1, match2)
                        # coupling = snn_matching(match1, match2)
                        # coupling = torch.from_numpy(coupling).to("cuda")
                    if self.match == "Random":
                        coupling = torch.full((x2_.shape[0], x2_.shape[0]), torch.tensor(1/x2_.shape[0]), device = "cuda")
                    #idx = torch.multinomial(coupling, num_samples = 1)
                    idx = torch.multinomial(torch.t(coupling), num_samples = 1)
                    x1_clone_1 = (x1_.clone())[torch.flatten(idx)].view(x1_.size())
                    idx = torch.multinomial(torch.t(coupling), num_samples = 1)
                    x1_clone_2 = (x1_.clone())[torch.flatten(idx)].view(x1_.size())
                    #x2_ = coupling @ x2_
                    #x2[subset] = x2_
                    x1[subset] = x1_clone_1
                    if self.unbiased: x1_clone[subset] = x1_clone_2

            if self.unbiased:
                self.trainer.datamodule.train_dataset = GEXADTDataset_Double(x1.to("cpu"), x1_clone.to("cpu"), x2.to("cpu"), y.to("cpu"))      
            else:
                self.trainer.datamodule.train_dataset = GEXADTDataset(x1.to("cpu"), x2.to("cpu"), y.to("cpu"))
        
    def on_train_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr = self.lr, weight_decay = self.wd)
        print(optimizer)
        print(filter(lambda p: p.requires_grad, self.parameters()))
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.lr, total_steps = self.trainer.max_epochs, pct_start = 0.1)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    