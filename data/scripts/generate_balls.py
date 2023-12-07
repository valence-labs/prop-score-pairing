# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import sys
import random
import argparse
import torch
import numpy as np
import pygame
from pygame import gfxdraw, init
from typing import Callable, Optional
from matplotlib import pyplot as plt
if "SDL_VIDEODRIVER" not in os.environ:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
COLOURS_ = [
    [2, 156, 154],
    [222, 100, 100],
    [149, 59, 123],
    [74, 114, 179],
    [27, 159, 119],
    [218, 95, 2],
    [117, 112, 180],
    [232, 41, 139],
    [102, 167, 30],
    [231, 172, 2],
    [167, 118, 29],
    [102, 102, 102],
]
SCREEN_DIM = 64
def square(
    x_,
    y_,
    surf,
    color=(204, 204, 0),
    radius=0.1,
    screen_width=SCREEN_DIM,
    y_shift=0.0,
    offset=None,
):
    if offset is None:
        offset = screen_width / 2
    scale = screen_width
    x = scale * x_ + offset
    y = scale * y_ + offset
    gfxdraw.box(
        surf, (int(x), int(y), int(radius * scale), int(radius * scale)), color,
    )
def circle(
    x_,
    y_,
    surf,
    color=(204, 204, 0),
    radius=0.1,
    screen_width=SCREEN_DIM,
    y_shift=0.0,
    offset=None,
):
    if offset is None:
        offset = screen_width / 2
    scale = screen_width
    x = scale * x_ + offset
    y = scale * y_ + offset
    gfxdraw.aacircle(
        surf, int(x), int(y - offset * y_shift), int(radius * scale), color
    )
    gfxdraw.filled_circle(
        surf, int(x), int(y - offset * y_shift), int(radius * scale), color
    )
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
class Balls(torch.utils.data.Dataset):
    ball_rad = 0.02
    screen_dim = 128
    def __init__(
        self,
        transform: Optional[Callable] = None,
        n_balls: int = 1,
        use_noise: bool = False,
    ):
        super(Balls, self).__init__()
        if transform is None:
            def transform(x):
                return x
        self.transform = transform
        pygame.init()
        self.screen1 = pygame.display.set_mode((self.screen_dim, self.screen_dim))
        self.surf1 = pygame.Surface((self.screen_dim, self.screen_dim))
        self.screen2 = pygame.display.set_mode((self.screen_dim, self.screen_dim))
        self.surf2 = pygame.Surface((self.screen_dim, self.screen_dim))
        self.n_balls = n_balls
        self.use_noise = use_noise
    def __len__(self) -> int:
        # arbitrary since examples are generated online.
        return 20000
    def draw_scene1(self, z):
        cmap_bg1 = plt.get_cmap("summer")
        self.surf1.fill((255, 255, 255))
        if z.ndim == 1:
            z = z.reshape((1, 2))
        for i in range(25):
            c = tuple(np.array(np.array(cmap_bg1(np.random.rand())) * 255, dtype=int)[:3])
            circle(
                np.random.rand(),
                np.random.rand(),
                self.surf1,
                color=c,
                radius=0.3,
                screen_width=self.screen_dim,
                y_shift=0.0,
                offset=0.0,
            )
        for i in range(z.shape[0]):
            circle(
                z[i, 0],
                z[i, 1],
                self.surf1,
                color=COLOURS_[i],
                radius=self.ball_rad,
                screen_width=self.screen_dim,
                y_shift=0.0,
                offset=0.0,
            )
        self.surf1 = pygame.transform.flip(self.surf1, False, True)
        self.screen1.blit(self.surf1, (0, 0))
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen1)), axes=(1, 0, 2)
        )
    def draw_scene2(self, z):
        cmap_bg2 = plt.get_cmap("Wistia")
        cmap = plt.get_cmap("viridis")
        theta = np.pi * 0.3
        U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        z -= 0.5
        z = z @ U
        z += 0.2
        z = sigmoid(2 * z)
        self.surf2.fill((255, 255, 255))
        for i in range(25):
            c = tuple(np.array(np.array(cmap_bg2(np.random.rand())) * 255, dtype=int)[:3])
            circle(
                np.random.rand(),
                np.random.rand(),
                self.surf2,
                color=c,
                radius=0.3,
                screen_width=self.screen_dim,
                y_shift=0.0,
                offset=0.0,
            )
        if z.ndim == 1:
            z = z.reshape((1, 2))
        for i in range(z.shape[0]):
            square(
            z[i, 0],
            z[i, 1],
            self.surf2,
            color=np.array(np.array(cmap(0.1 + i / self.n_balls)) * 256, dtype=int)[:3],#COLOURS_[i],
            radius=self.ball_rad * 2,
            screen_width=self.screen_dim,
            y_shift=0.0,
            offset=0.0,
        )
        self.surf2 = pygame.transform.flip(self.surf2, False, True)
        self.screen2.blit(self.surf2, (0, 0))
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen2)), axes=(1, 0, 2)
        )
    def __getitem__(self, item):
        raise NotImplemented()
class BlockOffset(Balls):
    def __init__(
        self,
        transform: Optional[Callable] = None,
        n_balls: int = 2,
        interventions_per_latent: int = 3,
        latent_case: str = 'iid',
        scm_mechanism: str = None,
        use_noise: bool = False,
    ):
        super().__init__(transform=transform, n_balls=n_balls, use_noise=use_noise)
        self.dataset_size= 20000
        self.latent_dim= self.n_balls*2
        self.intervention_case= 0
        self.intervene_all_latent= 1
        self.interventions_per_latent= interventions_per_latent
        self.latent_case= latent_case
        self.scm_mechanism= scm_mechanism
        
    def __getitem__(self, item):
        if self.latent_case == 'iid':
            z = self.get_observational_data_iid()
        elif self.latent_case == 'scm':
            z = self.get_observational_data_scm()
        else:
            print('Latent type not supported')
            sys.exit()
            
        y = -1*np.ones(2)
        
        if self.intervention_case:
            z= z.flatten()
            
            if self.intervene_all_latent:
                intervene_idx = np.random.randint(self.latent_dim, size=1)         
            else:
                intervene_idx = 0
            
            if self.interventions_per_latent > 1:
                intervene_range = np.linspace(0.25, 0.75, num= self.interventions_per_latent)
                intervene_val = [np.random.choice(intervene_range)]
            elif self.interventions_per_latent == 1:
                intervene_val = [0.5]
            
            if self.latent_case == 'iid':            
                z = self.get_interventional_data_iid(z, intervene_idx, intervene_val)
            elif self.latent_case == 'scm':
                z = self.get_interventional_data_scm(z, intervene_idx, intervene_val)
            else:
                print('Latent type not supported')
                sys.exit()
            
            y[0] = intervene_idx
            y[1] = intervene_val[0]
            
        x1, x2 = self.draw_scene1(z),self.draw_scene2(z)
        x1 = self.transform(x1)
        x2 = self.transform(x2)
        
        return z.flatten(), y, x1, x2
    
    def get_observational_data_iid(self):
        z = np.random.uniform(0.1, 0.9, size=(self.n_balls, 2))
        return z
    
    def get_interventional_data_iid(self, z, intervene_idx, intervene_val):
        z[intervene_idx] = intervene_val
        z = np.reshape(z, (self.n_balls, 2))        
        return z
    
    def get_observational_data_scm(self, x1=None, y1=None):
        
        if x1 is None:
            x1= np.random.uniform(0.1, 0.9, size=1)[0]
        
        if y1 is None:
            y1= np.random.uniform(0.1, 0.9, size=1)[0]
        
        if self.scm_mechanism == 'linear':
            constraint= x1+y1
        elif self.scm_mechanism == 'non_linear':
            constraint= 1.25 * (x1**2 + y1**2)
        
        if constraint >= 1.0:
            x2= np.random.uniform(0.1, 0.5, size=1)[0]
            y2= np.random.uniform(0.5, 0.9, size=1)[0]
        else:
            x2= np.random.uniform(0.5, 0.9, size=1)[0]
            y2= np.random.uniform(0.1, 0.5, size=1)[0]
            
        z= np.array([[x1, y1], [x2, y2]])
        
        return z
    
    def get_interventional_data_scm(self, z, intervene_idx, intervene_val):
        
        # Internvetion on the child Ball (Ball 2)
        if intervene_idx in [2, 3]:
            z[intervene_idx]= intervene_val
            z= np.reshape(z, (self.n_balls, 2))
            
        # Internvetion on the parent Ball (Ball 1)
        elif intervene_idx == 0:
            z= self.get_observational_data_scm(x1= intervene_val[0], y1= z[1])
        elif intervene_idx == 1:
            z= self.get_observational_data_scm(x1= z[0], y1= intervene_val[0])
        
        return z
    
# Input Parsing
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1,
                        help='')
    parser.add_argument('--train_size', type=int, default=20000,
                        help='')
    parser.add_argument('--test_size', type=int, default=20000,
                        help='')
    parser.add_argument('--distribution_case', type=str, default='observation', 
                    help='observation; intervention')
    parser.add_argument('--latent_case', type=str, default='iid', 
                    help='iid; scm')
    parser.add_argument('--scm_mechanism', type=str, default='linear', 
                    help='linear; non_linear')
    parser.add_argument('--interventions_per_latent', type= int, default=3,
                    help='')
    parser.add_argument('--use_noise', action='store_true', default=False,
                    help='Add background circles to the images')
    args = parser.parse_args()

    os.chdir("..")

    seed= args.seed
    train_size= args.train_size
    test_size= args.test_size
    distribution_case= args.distribution_case
    latent_case= args.latent_case
    scm_mechanism= args.scm_mechanism
    interventions_per_latent= args.interventions_per_latent
    #Random Seed
    random.seed(seed*10)
    np.random.seed(seed*10) 
    data_obj= BlockOffset(interventions_per_latent= interventions_per_latent, latent_case= latent_case, scm_mechanism= scm_mechanism, use_noise= args.use_noise)
    if distribution_case == 'observation':
        data_obj.intervention_case = 0
    elif distribution_case == 'intervention':
        data_obj.intervention_case = 1
    for data_case in ['train', 'val', 'test']: 
        base_dir= 'datasets/noisyballs_' + latent_case + '_' + scm_mechanism  + '/' + str(distribution_case)  + '/'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)        
        if os.path.exists(base_dir + data_case + '_' + 'x1' + '.npy') and os.path.exists(base_dir + data_case + '_' + 'x2' + '.npy') and os.path.exists(base_dir + data_case + '_' + 'y' + '.npy') and os.path.exists(base_dir + data_case + '_' + 'z' + '.npy'):
            print(f"Distribution case: {distribution_case}, data case: {data_case} already exists in {base_dir}.")
            continue 
        print('Distribution Case: ', distribution_case, 'Data Case: ', data_case)
        if data_case == 'train':
            dataset_size= args.train_size
        if data_case == 'val':
            dataset_size= int(args.train_size/4)
        elif data_case == 'test':
            dataset_size= args.test_size
        count=0
        final_z= []
        final_y= []
        final_x1= []
        final_x2= []
        for batch_idx, (z, y, x1, x2) in enumerate(data_obj):
            z= np.expand_dims(z, axis= 0)
            y= np.expand_dims(y, axis= 0)
            x1= np.expand_dims(x1, axis= 0)            
            x2= np.expand_dims(x2, axis= 0)            
            final_z.append(z)
            final_y.append(y)
            final_x1.append(x1)
            final_x2.append(x2)
            count+=1        
            if count >= dataset_size:
                break
        final_z= np.concatenate(final_z, axis=0)    
        final_y= np.concatenate(final_y, axis=0)    
        final_x1= np.concatenate(final_x1, axis=0)
        final_x2= np.concatenate(final_x2, axis=0)
        print(final_z.shape, final_y.shape, final_x1.shape, final_x2.shape)  
        print(final_z[:5])
        print(final_y[:5])
        f= base_dir + data_case + '_' + 'x1' + '.npy'
        np.save(f, final_x1)
        f= base_dir + data_case + '_' + 'x2' + '.npy'
        np.save(f, final_x2)
        f= base_dir + data_case + '_' + 'z' + '.npy'
        np.save(f, final_z)
        f= base_dir + data_case + '_' + 'y' + '.npy'
        np.save(f, final_y)
if __name__ == "__main__":
    main()