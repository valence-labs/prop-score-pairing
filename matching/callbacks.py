import numpy as np
from pytorch_lightning.callbacks import Callback
from .utils import scot_matching, snn_matching, eot_matching, latent_matching_score
from .models.classifier import BallsClassifier
from torchvision import transforms
import torch
from tqdm import tqdm
from timeit import default_timer as timer

MU = np.array([0.9906, 0.9902, 0.9922])
SIG = np.array([0.008, 0.008, 0.008])

class MatchingMetrics(Callback):
    def __init__(self, run_scot = False, eval_inv_factor = 2, eval_max_samples = 2500):
        self.run_scot = run_scot
        self.eval_inv_factor = eval_inv_factor ## eval on len(data_group)//eval_inv_factor number of samples within each group, int >= 1
        self.eval_max_samples = eval_max_samples ## max number of samples for evaluation --- eot might be too slow otherwise 

    def setup(self, trainer, pl_module, stage):
        self.train_loader = torch.utils.data.DataLoader(trainer.datamodule.train_dataset, batch_size = len(trainer.datamodule.train_dataset))
        
        if isinstance(pl_module, BallsClassifier):
            self.x1, self.x2, self.y, self.z = next(iter(self.train_loader))  ## only have ground truth z for balls dataset
            self.z, self.y = self.z.cpu().detach().numpy(), self.y.cpu().detach().numpy()
        else:
            self.x1, self.x2, self.y = next(iter(self.train_loader))
            self.y = self.y.cpu().detach().numpy()
        if self.run_scot:
            ## x1, x2 should be reshaped to (n, d)
            trace_avg = 0
            if isinstance(pl_module, BallsClassifier): mse_avg = 0
            for label in np.unique(self.y):
                subset = self.y == label 
                x1_lab, x2_lab = self.x1[subset], self.x2[subset]
                n_samples = min(len(x1_lab)//self.eval_inv_factor, self.eval_max_samples)
                x1_lab, x2_lab = x1_lab[range(n_samples)], x2_lab[[range(n_samples)]]
                x1_lab, x2_lab = x1_lab.reshape((x1_lab.shape[0], -1)), x2_lab.reshape((x2_lab.shape[0], -1))
                start_scot = timer()
                scot_coupling = scot_matching(x1_lab, x2_lab)
                end_scot = timer()
                print(f"scot took {end_scot - start_scot} seconds on label {label} with {len(x1_lab)} samples")
                trace = np.trace(scot_coupling)
                pl_module.logger.experiment.summary[f"SCOT Trace {label}"] = trace
                trace_avg += trace/len(np.unique(self.y))
                if isinstance(pl_module, BallsClassifier):
                    z_subset = self.z[subset]
		    z_subset = z_subset[range(n_samples)]
                    mse = latent_matching_score(scot_coupling, z_subset)
                    pl_module.logger.experiment.summary[f"SCOT MSE {label}"] = mse
                    mse_avg += mse/len(np.unique(self.y))
            pl_module.logger.experiment.summary["SCOT Trace Average"] = trace_avg
            if isinstance(pl_module, BallsClassifier): pl_module.logger.experiment.summary["SCOT MSE Average"] = mse_avg

        
    def on_train_epoch_end(self, trainer, pl_module):
        ## compute PS metrics
        if trainer.current_epoch % 5 == 0:
            trace_avg_ps_nn, trace_avg_ps_eot = 0, 0
            if isinstance(pl_module, BallsClassifier): mse_avg_ps_nn, mse_avg_ps_eot = 0, 0
            for label in np.unique(self.y):
                subset = self.y == label
                with torch.inference_mode():
                    x1_, x2_ = self.x1[subset].to("cuda"), self.x2[subset].to("cuda")
                    n_samples = min(len(x1_)//self.eval_inv_factor, self.eval_max_samples)
                    x1_, x2_ = x1_[range(n_samples)], x2_[range(n_samples)]
                    match_x1, match_x2 = pl_module(x1_, x2_)
                    match_x1, match_x2 = match_x1.cpu().detach().numpy(), match_x2.cpu().detach().numpy()
                start_snn = timer()
                ps_nn_coupling = snn_matching(match_x1, match_x2)
                end_snn = timer()
                print(f"kNN took {end_snn - start_snn} seconds on label {label} with {len(match_x1)} samples")
                ps_nn_trace = np.trace(ps_nn_coupling)
                self.log(f"PS + kNN Trace {label}",ps_nn_trace)
                start_eot = timer()
                ps_eot_coupling = eot_matching(match_x1, match_x2, max_iter=1000, verbose = False, use_sinkhorn_log = True)
                end_eot = timer()
                print(f"EOT took {end_eot - start_eot} seconds on label {label} with {len(match_x1)} samples")
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



