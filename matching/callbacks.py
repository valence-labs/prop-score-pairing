import numpy as np
from pytorch_lightning.callbacks import Callback
from .utils import scot_matching, snn_matching, eot_matching, latent_matching_score
from .models.classifier import BallsClassifier
from torchvision import transforms
import torch
from tqdm import tqdm

MU = np.array([0.9906, 0.9902, 0.9922])
SIG = np.array([0.008, 0.008, 0.008])

class MatchingMetrics(Callback):
    def __init__(self, run_scot = False):
        self.run_scot = run_scot

    def setup(self, trainer, pl_module, stage):
        self.full_loader = torch.utils.data.DataLoader(trainer.datamodule.full_dataset, batch_size = len(trainer.datamodule.full_dataset)//32)
        if isinstance(pl_module, BallsClassifier):
            self.x1, self.x2, self.y, self.z = next(iter(self.full_loader))  ## only have ground truth z for balls dataset
            self.z, self.y = self.z.cpu().detach().numpy(), self.y.cpu().detach().numpy()
        else:
            self.x1, self.x2, self.y = next(iter(self.full_loader))
            self.y = self.y.cpu().detach().numpy()
        if self.run_scot:
            ## x1, x2 should be reshaped to (n, d)
            trace_avg = 0
            if isinstance(pl_module, BallsClassifier): mse_avg = 0
            for label in np.unique(self.y):
                subset = self.y == label 
                x1_lab, x2_lab = self.x1[subset], self.x2[subset]
                x1_lab, x2_lab = x1_lab.reshape((x1_lab.shape[0], -1)), x2_lab.reshape((x2_lab.shape[0], -1))
                scot_coupling = scot_matching(x1_lab, x2_lab)
                trace = np.trace(scot_coupling)
                pl_module.logger.experiment.add_scalar(f"SCOT Trace {label}", trace)
                trace_avg += trace/len(np.unique(self.y))
                if isinstance(pl_module, BallsClassifier):
                    z_subset = self.z[subset]
                    mse = latent_matching_score(scot_coupling, z_subset)
                    pl_module.logger.experiment.add_scalar(f"SCOT MSE {label}", mse)
                    mse_avg += mse/len(np.unique(self.y))
            pl_module.logger.experiment.add_scalar("SCOT Trace Average", trace_avg)
            if isinstance(pl_module, BallsClassifier): pl_module.logger.experiment.add_scalar("SCOT MSE Average", mse_avg)

        
    def on_train_epoch_end(self, trainer, pl_module):
        ## compute PS metrics
        trace_avg_ps_nn, trace_avg_ps_eot = 0, 0
        if isinstance(pl_module, BallsClassifier): mse_avg_ps_nn, mse_avg_ps_eot = 0, 0
        for label in tqdm(np.unique(self.y)):
            subset = self.y == label
            with torch.no_grad():
                x1_, x2_ = self.x1[subset].to("cuda"), self.x2[subset].to("cuda")
                match_x1, match_x2 = pl_module(x1_, x2_)
                match_x1, match_x2 = match_x1.cpu().detach().numpy(), match_x2.cpu().detach().numpy()
            ps_nn_coupling = snn_matching(match_x1, match_x2)
            ps_nn_trace = np.trace(ps_nn_coupling)
            self.log(f"PS + kNN Trace {label}",ps_nn_trace)
            ps_eot_coupling = eot_matching(match_x1, match_x2, max_iter=1000, verbose = False)
            ps_eot_trace = np.trace(ps_eot_coupling)
            self.log(f"PS + EOT Trace {label}", ps_eot_trace)
            trace_avg_ps_nn += ps_nn_trace/len(np.unique(self.y))
            trace_avg_ps_eot += ps_eot_trace/len(np.unique(self.y))

            if isinstance(pl_module, BallsClassifier):
                z_subset = self.z[subset]
                mse_ps_nn = latent_matching_score(ps_nn_coupling, z_subset)
                mse_ps_eot = latent_matching_score(ps_eot_coupling, z_subset)
                pl_module.log(f"PS + kNN MSE {label}", mse_ps_nn)
                pl_module.log(f"PS + EOT MSE {label}", mse_ps_eot)

                mse_avg_ps_nn += mse_ps_nn/len(np.unique(self.y))
                mse_avg_ps_eot += mse_ps_eot/len(np.unique(self.y))

        pl_module.log("PS + kNN Trace Average", trace_avg_ps_nn)
        pl_module.log("PS + EOT Trace Average", trace_avg_ps_eot)
        if isinstance(pl_module, BallsClassifier): 
            pl_module.log("PS + kNN MSE Average", mse_avg_ps_nn)
            pl_module.log("PS + EOT MSE Average", mse_avg_ps_eot)
            pl_module.log("Random MSE", ((np.random.permutation(self.z) - self.z)**2).mean())



