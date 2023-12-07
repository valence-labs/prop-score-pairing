import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from .. import convert_to_labels
from typing import Tuple, Optional


## Normalization statistics, channel means and sd

## These are for old data with no background, largely unused

MU = np.array([0.998, 0.998, 0.998])
SIG = np.array([0.034, 0.025, 0.025])

## View/"Modality" 1

MU_noisy_1 = np.array([0.50454001, 0.75075871, 0.40544085])
SIG_noisy_1 = np.array([0.29195627, 0.14611416, 0.05556526])

## View/"Modality" 2

MU_noisy_2 = np.array([0.98185011, 0.75588502, 0.09406329])
SIG_noisy_2 = np.array([0.05103349, 0.15451756, 0.15762656])

class BallsDataset(Dataset):
    def __init__(self, 
                 x1: torch.Tensor, 
                 x2: torch.Tensor, 
                 y: torch.Tensor, 
                 z: torch.Tensor):
        super().__init__()
        self.x1 = x1
        self.x2 = x2
        self.z = z
        ### convert y to labels
        self.y = convert_to_labels(y)
        ###
        self.transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MU,
                std=SIG,
            ),
        ]
        )
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x1, x2, y, z = self.x1[index], self.x2[index], torch.tensor(self.y[index]).long(), torch.tensor(self.z[index]).float().flatten()
        x1, x2 = self.transform(x1), self.transform(x2)

        return x1, x2, y, z
    
class NoisyBallsDataset(BallsDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transform1 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MU_noisy_1,  
                std=SIG_noisy_1,
            ),
        ]
        )
        self.transform2 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MU_noisy_2,  
                std=SIG_noisy_2,
            ),
        ]
        )

        def __getitem__(self, index):
            x1, x2, y, z = self.x1[index], self.x2[index], torch.tensor(self.y[index]).long(), torch.tensor(self.z[index]).float().flatten()
            x1, x2 = self.transform1(x1), self.transform2(x2)

            return x1, x2, y, z
class BallsDataModule(LightningDataModule):
    """
    Modified from https://github.com/facebookresearch/CausalRepID/blob/main/data/balls_dataset_loader.py
    Interventional Causal Representation Learning, Ahuja et al, https://arxiv.org/abs/2209.11924
    """
    def __init__(self,
                 batch_size: int = 100):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = BallsDataset
    def prepare_data(self):
        self.data_dir = "/mnt/ps/home/CORP/johnny.xi/sandbox/matching/data/datasets/balls_scm_non_linear/intervention/"
    def setup(self, stage: str):
        self.x1_tr = np.load(self.data_dir +  'train_' + 'x1' + '.npy')
        self.x2_tr = np.load(self.data_dir +  'train_' + 'x2' + '.npy')
        self.z_tr = np.load(self.data_dir +  'train_' + 'z' + '.npy')
        self.y_tr = np.load(self.data_dir +  'train_' + 'y' + '.npy')  
        self.x1_val = np.load(self.data_dir +  'val_' + 'x1' + '.npy')
        self.x2_val = np.load(self.data_dir +  'val_' + 'x2' + '.npy')
        self.z_val = np.load(self.data_dir +  'val_' + 'z' + '.npy')
        self.y_val = np.load(self.data_dir +  'val_' + 'y' + '.npy')  
        self.x1_test = np.load(self.data_dir +  'test_' + 'x1' + '.npy')
        self.x2_test = np.load(self.data_dir +  'test_' + 'x2' + '.npy')
        self.z_test = np.load(self.data_dir +  'test_' + 'z' + '.npy')
        self.y_test = np.load(self.data_dir +  'test_' + 'y' + '.npy')  
        self.labels = convert_to_labels(self.y_tr)
        self.train_dataset = self.dataset(x1 = self.x1_tr, x2 = self.x2_tr, y = self.y_tr, z = self.z_tr)
        if stage == "test":
            self.val_dataset = self.dataset(x1 = self.x1_val, x2 = self.x2_val, y = self.y_val, z = self.z_val)
            self.test_dataset = self.dataset(x1 = self.x1_test, x2 = self.x2_test, y = self.y_test, z = self.z_test)
    def train_dataloader(self):
        return DataLoader(self.dataset(x1 = self.x1_tr, x2 = self.x2_tr, y = self.y_tr, z = self.z_tr), batch_size = self.batch_size, num_workers=8)
    def val_dataloader(self):
        return DataLoader(self.dataset(x1 = self.x1_val, x2 = self.x2_val, y = self.y_val, z = self.z_val), batch_size = self.batch_size, num_workers=8)
    def test_dataloader(self):
        return DataLoader(self.dataset(x1 = self.x1_test, x2 = self.x2_test, y = self.y_test, z = self.z_test), batch_size = self.batch_size, num_workers=8)

class NoisyBallsDataModule(BallsDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = NoisyBallsDataset
    def prepare_data(self):
        self.data_dir = "/mnt/ps/home/CORP/johnny.xi/sandbox/matching/data/datasets/noisyballs_scm_non_linear/intervention/"

class GEXADTDataModule(LightningDataModule):
    def __init__(self,
        batch_size: int = 256,
        d1_sub: bool = False):

        super().__init__()
        self.batch_size = batch_size
        self.d1_sub = d1_sub ## Give option to subset the data to donor 1 

    def _train_val_split_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if "split" not in df.columns:
            raise KeyError(f"Missing column 'split' from dataframe, got: {df.columns}")
        train_idx = df["split"] == "train"
        val_idx = df["split"] == "val"
        test_idx = df["split"] == "test" 

        if min(np.sum(train_idx), np.sum(val_idx), np.sum(test_idx)) > 0.01*len(train_idx): ## If each split is at least 1% of full data, use the given split
            train_df = df[train_idx].reset_index()
            val_df = df[val_idx].reset_index()
            test_df = df[test_idx].reset_index()
        else:                                                                               
            df = df.reset_index()
            train_df = df[:round(len(df)*0.8)].reset_index()
            val_df = df[round(len(df)*0.8):round(len(df)*0.9)].reset_index()
            test_df = df[round(len(df)*0.9):].reset_index()

        return train_df, val_df, test_df

    def load_data(self) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        data_adt = pd.read_parquet("/mnt/ps/home/CORP/johnny.xi/sandbox/matching/data/datasets/neurips_2021_bm/adt.parquet") 
        data_gex = pd.read_parquet("/mnt/ps/home/CORP/johnny.xi/sandbox/matching/data/datasets/neurips_2021_bm/gex_pca_200.parquet") 

        if self.d1_sub:
            d1 = ["s1d1", "s1d2", "s1d3"]
            data_adt = data_adt.loc[data_adt.batch.isin(d1)]
            data_gex = data_gex.loc[data_gex.batch.isin(d1)]
            data_adt.CT_id = data_adt.cell_type.cat.remove_unused_categories().cat.codes
            data_gex.CT_id = data_gex.cell_type.cat.remove_unused_categories().cat.codes

        return self._train_val_split_df(data_adt), self._train_val_split_df(data_gex)
    
    def df_to_torch(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
         df.columns = df.columns.astype(str)
         label_tensor = torch.tensor(df["CT_id"]).long()
         dat = df.filter(regex="^[0-9]").astype("float32").values  ## the data columns starts with numeric and metadata is non-numeric
         dat_tensor = torch.from_numpy(dat).float()

         return dat_tensor, label_tensor


    def setup(self, stage: str) -> None:

        (train_df_adt, val_df_adt, test_df_adt), (train_df_gex, val_df_gex, test_df_gex)  = self.load_data()

        self.train_data_adt, self.train_labels = self.df_to_torch(train_df_adt)
        self.val_data_adt, self.val_labels = self.df_to_torch(val_df_adt)
        self.test_data_adt, self.test_labels = self.df_to_torch(test_df_adt)

        self.train_data_gex, _ = self.df_to_torch(train_df_gex)
        self.val_data_gex, _ = self.df_to_torch(val_df_gex)
        self.test_data_gex, _ = self.df_to_torch(test_df_gex)

        self.train_dataset = GEXADTDataset(self.train_data_adt, self.train_data_gex, self.train_labels)
        self.val_dataset = GEXADTDataset(self.val_data_adt, self.val_data_gex, self.val_labels)
        self.test_dataset = GEXADTDataset(self.test_data_adt, self.test_data_gex, self.test_labels)   

        self.labels = self.train_labels ## for use during setup of module

    
    def train_dataloader(self) -> DataLoader:
        print("train dataloader")
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers=8)
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=8)
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers=8)
class GEXADTDataset(Dataset):
    def __init__(self, 
                 data_adt: torch.Tensor, 
                 data_gex: torch.Tensor, 
                 labels: torch.Tensor):
         super().__init__()
         self.adt_tensor = data_adt
         self.gex_tensor = data_gex
         self.label_tensor = labels

    def __len__(self):
         return len(self.adt_tensor)

    def __getitem__(self, index: int):
        adt_row = self.adt_tensor[index,:]
        gex_row = self.gex_tensor[index,:]
        lab = self.label_tensor[index]

        return adt_row, gex_row, lab ## x1, x2, y

class GEXADTDataset_Double(GEXADTDataset):
    """
    Loading two samples from GEX modality for 2SLS loss
    """
    def __init__(self, 
                 data_adt: torch.Tensor, 
                 data_adt_2: torch.Tensor, 
                 data_gex: torch.Tensor, 
                 labels: torch.Tensor):
         super().__init__(data_adt, data_gex, labels)
         self.adt_tensor_2 = data_adt_2

    def __getitem__(self, index: int):
        adt_row = self.adt_tensor[index,:]
        adt_row_2 = self.adt_tensor_2[index,:]
        gex_row_2 = self.gex_tensor[index,:]
        lab = self.label_tensor[index]

        return adt_row, adt_row_2, gex_row_2, lab ## x1, x1(2), x2, y
     


