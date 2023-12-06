### Training

Example training script:

```
python main.py --dataset=GEXADT --max_epochs=250 --batch_size=256 --eval_interval=1 --run_scot
```

The command takes the following possible parameters:

- `--max_epochs`
- `--batch_size`
- `--lr`
- `--dataset`: One of "GEXADT", for CITE-seq data, or "BALLS", for the synthetic interventional balls data.
- `--model`: One of "CLASSIFIER" or "VAE".
- `--eval_interval`: How often to evaluate the model by computing matching metrics. Default is set to evaluate at every epoch (`= 1`). If the matching metrics are too slow, you may want to make this larger.
- `--run_scot`: Boolean flag for whether to run Gromov-Wasserstein OT (SCOT) and store its metrics prior to training. This can be slow for larger datasets!

By default, `main.py` will train a classifier on the CITE-seq data for 50 epochs, with batch size 256, and with learning rate 0.0001.  

There are also parameters related to the VAE loss from:
Yang, Karren Dai, et al. "Multi-domain translation between single-cell imaging and sequencing data using autoencoders." Nature communications 12.1 (2021): 31.

- `--lamb`
- `--alpha`
- `--beta`

### Inference

The training script will store a checkpoint with the best validation loss at `results/checkpoints/*.ckpt`. With this checkpoint, we can evaluate final matching metrics on the test set with the following example script:  

```
python3 inference.py --checkpoint=results/checkpoints/PS-epoch=06-full_val_loss=0.63.ckpt --dataset=GEXADT
```

The result will be stored under `results/`, in a CSV file.

The command takes the following parameters: 

- `--checkpoint`
- `--dataset`
- `--run_scot`

Besides `--checkpoint`, which is self-explanatory, the remaining parameters are identical to `main.py`.

### Probing

Another downstream task is to probe whether the matched samples resulting from a trained classifier/VAE are able to be used for cross-modality prediction. The following script takes a checkpoint and trains a 2-layer MLP (which we call a probe) minimizing MSE for prediction. 

__Note__: this is currently only supported for the CITE-seq data, as image translation would require a much more sophisticated architecture.

```
python3 probing.py --checkpoint=results/checkpoints/PS-epoch=06-full_val_loss=0.63.ckpt
```

This command takes the following parameters, note the training parameters are for training the MLP probe.  

- `--checkpoint`: This can be either a checkpoint to a VAE or classifier, or the string "gt", for ground truth matching, or "random", for a within-group random permutation.
- `--max_epochs`
- `--batch_size`
- `--lr`
- `--unbiased`: Boolean flag for whether to use the unbiased 2-stage least squares loss (this involves sampling twice from the source modality). 

