import sys
import numpy as np
from pytorch_lightning.callbacks import Callback
from .utils import scot_matching, snn_matching, eot_matching, latent_matching_score, compute_avg_FOSCTTM
from .models.classifier import BallsClassifier
from .data_utils.datamodules import BallsDataModule, GEXADTDataModule

from torchvision import transforms
import torch
from tqdm import tqdm
from timeit import default_timer as timer

MU = np.array([0.9906, 0.9902, 0.9922])
SIG = np.array([0.008, 0.008, 0.008])

def compute_metrics(match1, match2, y, matching, data, z = None, embedding = None):
    traces = []
    foscttms = []
    outputs = {}
    if isinstance(data, BallsDataModule): mses, random_mses = [], []
    for label in np.unique(y):
        subset = y == label 
        if embedding is not None:
            x1_, x2_ = embedding(match1[subset].to("cuda"), match2[subset].to("cuda"))
        else:
            x1_, x2_ = match1[subset], match2[subset]
        x1_, x2_ = x1_.reshape((x1_.shape[0], -1)).cpu().detach().numpy(), x2_.reshape((x2_.shape[0], -1)).cpu().detach().numpy()
        print(f"starting matching on label {label} with {len(x1_)} samples...")
        start = timer()
        coupling = matching(x1_, x2_)
        if isinstance(coupling, torch.Tensor): coupling = coupling.cpu().detach().numpy()
        end = timer()
        print(f"{matching} took {end - start} seconds on label {label} with {len(x1_)} samples")
        if np.isnan(coupling).any(): continue ## skip everything if any nans 

        trace = np.trace(coupling)/len(x1_)
        traces.append(trace)
        outputs[f"Trace {label}"] = trace
        x2_matched = coupling @ x2_

        if isinstance(data, GEXADTDataModule):
            FOSCTTM = compute_avg_FOSCTTM(x2_, x2_matched)
            foscttms.append(FOSCTTM)
            outputs[f"FOSCTTM {label}"] = float(FOSCTTM)
        if isinstance(data, BallsDataModule):
            z_subset = z[subset]
            z_subset = z_subset.cpu().detach().numpy()
            mse = latent_matching_score(coupling, z_subset)
            random_mse = float(((np.random.permutation(z_subset) - z_subset)**2).mean())
            mses.append(mse)
            random_mses.append(random_mse)
            outputs[f"MSE {label}"] = float(mse)
            outputs[f"Random MSE {label}"] = float(random_mse)
    
    outputs["Average Trace"] = np.mean(traces)
    if isinstance(data, GEXADTDataModule): outputs["Average FOSCTTM"] = np.mean(foscttms)
    if isinstance(data, BallsDataModule):
        outputs["Average MSE"] = np.mean(mses)
        outputs["Average Random MSE"] = np.mean(random_mses)

    return outputs


class MatchingMetrics(Callback):
    def __init__(self, run_scot = False, eval_inv_factor = 1, eval_max_samples = 2500, eval_interval = 1):
        self.run_scot = run_scot
        self.eval_inv_factor = eval_inv_factor ## eval on len(data_group)//eval_inv_factor number of samples within each group, int >= 1
        self.eval_max_samples = eval_max_samples ## max number of samples for evaluation --- eot might be too slow otherwise 
        self.eval_interval = eval_interval
    def setup(self, trainer, pl_module, stage):
        self.val_loader = torch.utils.data.DataLoader(trainer.datamodule.val_dataset, batch_size = len(trainer.datamodule.val_dataset))
        if isinstance(trainer.datamodule, BallsDataModule):
            self.x1, self.x2, self.y, self.z = next(iter(self.val_loader))  ## only have ground truth z for balls dataset
            self.z, self.y = self.z.cpu().detach().numpy(), self.y.cpu().detach().numpy()
        else:
            self.x1, self.x2, self.y = next(iter(self.val_loader))
            self.y = self.y.cpu().detach().numpy()
            self.z = None
        if self.run_scot:

            outputs_SCOT = compute_metrics(self.x1, self.x2, self.y, scot_matching, trainer.datamodule, self.z)
            for metric in outputs_SCOT:
                    if str(metric).split()[0] == "Average":
                        pl_module.logger.experiment.summary["SCOT" + metric] = outputs_SCOT[metric]
            
    def on_train_epoch_end(self, trainer, pl_module):
        ## compute PS metrics
        if (trainer.current_epoch + 1) % self.eval_interval == 0:
            with torch.no_grad():
                outputs_EOT = compute_metrics(self.x1, self.x2, self.y, eot_matching, trainer.datamodule, self.z, pl_module.forward)
                outputs_SNN = compute_metrics(self.x1, self.x2, self.y, snn_matching, trainer.datamodule, self.z, pl_module.forward)
                for metric in outputs_EOT:
                    if str(metric).split()[0] == "Average":
                        pl_module.log("EOT" + metric, outputs_EOT[metric])
                        pl_module.log("SNN" + metric, outputs_SNN[metric])

          