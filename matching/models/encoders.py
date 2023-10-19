import torch
from torch import nn
from torchvision.models import resnet18, resnet50, resnet34

class ImageEncoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(ImageEncoder, self).__init__()        
        
        self.latent_dim = latent_dim
        self.width = latent_dim * 2
        
        self.base_model = resnet18(pretrained=True)        
        self.feat_layers= list(self.base_model.children())[:-1]
        self.feat_net= nn.Sequential(*self.feat_layers)
        
        self.fc_layers= [                    
                    nn.Linear(512, self.width),
                    nn.LeakyReLU(),
                    nn.Linear(self.width, self.latent_dim),
                ] 
        self.fc_net = nn.Sequential(*self.fc_layers)
        
    def forward(self, x):
        x= self.feat_net(x)
        x= x.view(x.shape[0], x.shape[1])
        x= self.fc_net(x)
        return x