import sys
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from .utils import scot_matching, snn_matching, eot_matching, latent_matching_score, compute_avg_FOSCTTM
from .models.classifier import BallsClassifier
from .data_utils.datamodules import BallsDataModule, GEXADTDataModule

from torchvision import transforms
import torch
from tqdm import tqdm
from timeit import default_timer as timer

from typing import Callable, Optional, Union, Dict, Any

MU = np.array([0.9906, 0.9902, 0.9922])
SIG = np.array([0.008, 0.008, 0.008])

def compute_metrics(match1: torch.Tensor, 
                    match2: torch.Tensor, 
                    y: torch.Tensor, 
                    data: Union[BallsDataModule, GEXADTDataModule], 
                    matching: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, 
                    z: Optional[torch.Tensor] = None, 
                    embedding: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None) -> Dict[str, Any]:
    traces = []
    foscttms = []
    outputs = {}
    if isinstance(data, BallsDataModule): mses, random_mses = [], []
    for label in torch.unique(y):
        subset = y == label 
        if embedding is not None:
            x1_, x2_ = embedding(match1[subset].to("cuda"), match2[subset].to("cuda"))
        else:
            x1_, x2_ = match1[subset], match2[subset]
        #x1_, x2_ = x1_.reshape((x1_.shape[0], -1)).cpu().detach().numpy(), x2_.reshape((x2_.shape[0], -1)).cpu().detach().numpy()
        print(f"starting matching on label {label} with {len(x1_)} samples...")
        start = timer()
        coupling = matching(x1_, x2_)
        end = timer()
        print(f"{matching} took {end - start} seconds on label {label} with {len(x1_)} samples")
        if torch.isnan(coupling).any(): continue ## skip everything if any nans 

        trace = float(torch.trace(coupling)/len(x1_))
        traces.append(trace)
        outputs[f"Trace {label}"] = trace
        #x2_matched = coupling @ x2_  ## projection
        idx = torch.multinomial((coupling), num_samples = 1)  ## can also do sampling
        x2_matched = (x2_.clone())[torch.flatten(idx)].view(x2_.size())

        if isinstance(data, GEXADTDataModule):
            FOSCTTM = compute_avg_FOSCTTM(x2_, x2_matched)
            foscttms.append(FOSCTTM)
            outputs[f"FOSCTTM {label}"] = float(FOSCTTM)
        if isinstance(data, BallsDataModule):
            z_subset = z[subset]
            mse = float(latent_matching_score(coupling, z_subset))
            random_mse = float(((z_subset[torch.randperm(z_subset.size()[0])] - z_subset)**2).mean())
            mses.append(mse)
            random_mses.append(random_mse)
            outputs[f"MSE {label}"] = float(mse)
            outputs[f"Random MSE {label}"] = float(random_mse)
    
    outputs["Average Trace"] = float(np.mean(traces))
    if isinstance(data, GEXADTDataModule): outputs["Average FOSCTTM"] = float(np.mean(foscttms))
    if isinstance(data, BallsDataModule):
        outputs["Average MSE"] = float(np.mean(mses))
        outputs["Average Random MSE"] = float(np.mean(random_mses))

    return outputs


class MatchingMetrics(Callback):
    def __init__(self, 
                run_scot: bool = False,
                eval_inv_factor: int = 1, 
                eval_max_samples: int = 2500, 
                eval_interval: int = 1) -> None:
        
        self.run_scot = run_scot
        self.eval_inv_factor = eval_inv_factor ## eval on len(data_group)//eval_inv_factor number of samples within each group, int >= 1
        self.eval_max_samples = eval_max_samples ## max number of samples for evaluation --- eot might be too slow otherwise 
        self.eval_interval = eval_interval

    def setup(self, 
              trainer: pl.Trainer, 
              pl_module: pl.LightningModule, 
              stage: Optional[str]) -> None:
        
        self.loader = torch.utils.data.DataLoader(trainer.datamodule.val_dataset, batch_size = len(trainer.datamodule.val_dataset))
        if isinstance(trainer.datamodule, BallsDataModule):
            self.x1, self.x2, self.y, self.z = next(iter(self.loader))  ## only have ground truth z for balls dataset
            self.z = self.z
        elif isinstance(trainer.datamodule, GEXADTDataModule):
            self.x1, self.x2, self.y = next(iter(self.loader))
            self.z = None
        if self.run_scot:
            outputs_SCOT = compute_metrics(match1 = self.x1, match2 = self.x2, y = self.y, matching = scot_matching, data = trainer.datamodule, z = self.z)
            for metric in outputs_SCOT:
                    if str(metric).split()[0] == "Average":
                        pl_module.logger.experiment.summary["SCOT" + metric] = outputs_SCOT[metric]
            
    def on_train_epoch_end(self, 
                           trainer: pl.Trainer, 
                           pl_module: pl.LightningModule) -> None:
        ## compute PS metrics
        if (trainer.current_epoch + 1) % self.eval_interval == 0:
            with torch.no_grad():
                outputs_EOT = compute_metrics(match1 = self.x1, match2 = self.x2, y = self.y, matching = eot_matching, data = trainer.datamodule, z = self.z, embedding = pl_module.forward)
                outputs_SNN = compute_metrics(match1 = self.x1, match2 = self.x2, y = self.y, matching = snn_matching, data = trainer.datamodule, z = self.z, embedding = pl_module.forward)
                for metric in outputs_EOT:
                    if str(metric).split()[0] == "Average":
                        pl_module.log("EOT" + metric, outputs_EOT[metric])
                        pl_module.log("SNN" + metric, outputs_SNN[metric])

          