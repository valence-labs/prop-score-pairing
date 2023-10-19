import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from .. import convert_to_labels

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

        self.full_dataset = BallsDataset(self.x1_tr, self.x2_tr, self.y_tr, self.z_tr)



    def train_dataloader(self):
        return DataLoader(BallsDataset(self.x1_tr, self.x2_tr, self.y_tr, self.z_tr), batch_size = self.batch_size)
    def val_dataloader(self):
        return DataLoader(BallsDataset(self.x1_val, self.x2_val, self.y_val, self.z_val), batch_size = self.batch_size)
    def test_dataloader(self):
        return DataLoader(BallsDataset(self.x1_test, self.x2_test, self.y_test, self.z_test), batch_size = self.batch_size)