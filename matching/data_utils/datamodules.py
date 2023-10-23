import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from .. import convert_to_labels
from typing import Tuple, Optional

class DomainSampler(BatchSampler):
    def __init__(
        self,
        metadata: pd.DataFrame,
        batch_size: int,
        random_seed: Optional[int] = None,
    ) -> None:
        self.batch_size = batch_size
        self.sub_batch_size = self.batch_size//12
        self.metadata = metadata
        self._rng = np.random.RandomState(random_seed)
        # grouping by domain
        self._groups = self.metadata.groupby("batch")
        self._len = sum((len(group) // self.sub_batch_size for _, group in self._groups)) // (
            self.batch_size // self.sub_batch_size
        )

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        # copy and shuffle (.sample) the groups into a new list
        # we only want the indices of the rows, do not need the whole dataframe
        groups = [group.sample(frac=1, random_state=self._rng).index for _, group in self._groups]
        self._rng.shuffle(groups)
        # prepare the batches by slicing samples out from each group
        sub_batches = []
        for group in groups:
            for i in range(len(group) // self.sub_batch_size):
                sub_batches.append(group[i * self.sub_batch_size : (i + 1) * self.sub_batch_size])
        # shuffle them so that groups are distributed randomly over the epoch, then yield batches
        self._rng.shuffle(sub_batches)
        for i in range(len(sub_batches) // (self.batch_size // self.sub_batch_size)):
            yield np.array(
                sub_batches[
                    i * (self.batch_size // self.sub_batch_size) : (i + 1) * (self.batch_size // self.sub_batch_size)
                ]
            ).reshape(-1)




MU = np.array([0.9906, 0.9902, 0.9922])
SIG = np.array([0.008, 0.008, 0.008])

class BallsDataset(Dataset):
    def __init__(self, x1, x2, y, z):
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
                mean=MU,  # channel means - the images are mostly white so close to 1.
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

class BallsDataModule(LightningDataModule):
    def __init__(self,batch_size):
        super().__init__()
        self.batch_size = batch_size
    def prepare_data(self):
        self.data_dir = "/mnt/ps/home/CORP/johnny.xi/sandbox/matching/data/datasets/balls_scm_non_linear/intervention/"
    def setup(self, stage:str):
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

        # self.x1_full = np.concatenate((self.x1_tr, self.x1_val, self.x1_test), axis = 0)
        # self.x2_full = np.concatenate((self.x2_tr, self.x2_val, self.x2_test), axis = 0)
        # self.y_full = np.concatenate((self.y_tr, self.y_val, self.y_test))
        # self.z_full = np.concatenate((self.z_tr, self.z_val, self.z_test))
        self.labels = convert_to_labels(self.y_tr)
        self.train_dataset = BallsDataset(self.x1_tr, self.x2_tr, self.y_tr, self.z_tr)

    def train_dataloader(self):
        return DataLoader(BallsDataset(self.x1_tr, self.x2_tr, self.y_tr, self.z_tr), batch_size = self.batch_size, num_workers=8)
    def val_dataloader(self):
        return DataLoader(BallsDataset(self.x1_val, self.x2_val, self.y_val, self.z_val), batch_size = self.batch_size, num_workers=8)
    def test_dataloader(self):
        return DataLoader(BallsDataset(self.x1_test, self.x2_test, self.y_test, self.z_test), batch_size = self.batch_size, num_workers=8)

class GEXADTDataModule(LightningDataModule):
    def __init__(self,
        batch_size: int,
        d1_sub: bool = False
        ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.d1_sub = d1_sub ## subset to donor 1?

    def _train_val_split_df(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if "split" not in df.columns:
            raise KeyError(f"Missing column 'split' from dataframe, got: {df.columns}")
        train_idx = df["split"] == "train"
        val_idx = df["split"] == "val"
        test_idx = df["split"] == "test" 
        if min(np.sum(train_idx), np.sum(val_idx), np.sum(test_idx)) > 0.01*len(train_idx): ## If each split is at least 1% of full data
            train_df = df[train_idx].reset_index()
            val_df = df[val_idx].reset_index()
            test_df = df[test_idx].reset_index()
        else:
            df = df.reset_index()
            train_df = df[:round(len(df)*0.8)].reset_index()
            val_df = df[round(len(df)*0.8):round(len(df)*0.9)].reset_index()
            test_df = df[:round(len(df)*0.9)].reset_index()
        return train_df, val_df, test_df

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data_adt = pd.read_parquet("/mnt/ps/home/CORP/johnny.xi/sandbox/matching/data/datasets/neurips_2021_bm/adt.parquet") ## gonna want to have two of these
        data_gex = pd.read_parquet("/mnt/ps/home/CORP/johnny.xi/sandbox/matching/data/datasets/neurips_2021_bm/gex_pca_200.parquet") ## gonna want to have two of these
        if self.d1_sub:
            d1 = ["s1d1", "s1d2", "s1d3"]
            data_adt = data_adt.loc[data_adt.batch.isin(d1)]
            data_gex = data_gex.loc[data_gex.batch.isin(d1)]
            ## reset codes
            data_adt.CT_id = data_adt.cell_type.cat.remove_unused_categories().cat.codes
            data_gex.CT_id = data_gex.cell_type.cat.remove_unused_categories().cat.codes

        return self._train_val_split_df(data_adt), self._train_val_split_df(data_gex)

    def setup(self, stage: Optional[str] = None) -> None:
        # see https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html#setup
        # this hook is called on every process when using DDP at the beginning of fit
        (train_df_adt, val_df_adt, test_df_adt), (train_df_gex, val_df_gex, test_df_gex)  = self.load_data()
        # unpack the prepared metadata
        self.train_data_adt = train_df_adt
        self.val_data_adt = val_df_adt
        self.test_data_adt = test_df_adt

        self.train_data_gex = train_df_gex
        self.val_data_gex = val_df_gex
        self.test_data_gex = test_df_gex

        self.train_dataset = GEXADTDataset(self.train_data_adt, self.train_data_gex)
        self.labels = torch.tensor(self.train_data_adt.CT_id).long()        
    
    def train_dataloader(self) -> DataLoader:
        train_dataset = GEXADTDataset(
            self.train_data_adt,
            self.train_data_gex            
            )
        return DataLoader(train_dataset, batch_sampler = DomainSampler(self.train_data_adt, batch_size = self.batch_size), num_workers = 8)

    def val_dataloader(self) -> DataLoader:
        val_dataset = GEXADTDataset(
            self.val_data_adt,
            self.val_data_gex            
            )
        return DataLoader(val_dataset, batch_sampler = DomainSampler(self.val_data_adt, batch_size = self.batch_size), num_workers = 8)

    def test_dataloader(self) -> DataLoader:
        test_dataset = GEXADTDataset(
            self.test_data_adt,
            self.vest_data_gex            
            )
        return DataLoader(test_dataset, batch_sampler = DomainSampler(self.test_data_adt, batch_size = self.batch_size), num_workers = 8)

class GEXADTDataset(Dataset):
    def __init__(self, data_adt, data_gex):
         super().__init__()
         self.adt = data_adt
         self.gex = data_gex

    def __len__(self):
         return len(self.adt)
    
    def process_row(self, row):
        label = torch.tensor(row["CT_id"]).long()
        dat = row.filter(regex="^[0-9]").astype("float32")
        dat_tensor = torch.from_numpy(dat.values).float()

        return dat_tensor, label

    def __getitem__(self, index: int):
        adt_row = self.adt.iloc[index]
        gex_row = self.gex.iloc[index]
        adt_dat, adt_lab = self.process_row(adt_row)
        gex_dat, gex_lab = self.process_row(gex_row)

        assert adt_lab == gex_lab, f"Label mismatch at index {index}, ensure modalities are matched!"

        return adt_dat, gex_dat, adt_lab ## x1, x2, y
     


