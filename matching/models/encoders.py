import torch
from torch import nn
from torchvision.models import resnet18, resnet50, resnet34

class ImageEncoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(ImageEncoder, self).__init__()        
        
        self.latent_dim = latent_dim

        ### This image encoder is from https://github.com/facebookresearch/CausalRepID

        # self.width = min(latent_dim * 4, 256)
        
        # self.base_model = resnet18(pretrained=True)        
        # self.feat_layers= list(self.base_model.children())[:-1]
        # self.feat_net= nn.Sequential(*self.feat_layers)

        # self.fc_layers= [                    
        #             nn.Linear(512, self.width),
        #             nn.LeakyReLU(),
        #             nn.Linear(self.width, self.latent_dim),
        #         ] 
        # self.fc_net = nn.Sequential(*self.fc_layers)

        ## Image encoder from https://github.com/uhlerlab/cross-modal-autoencoders

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
        in_channels = 3
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.feat_net = nn.Sequential(*modules)
        self.fc_net = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)
        

        
    def forward(self, x):
        x= self.feat_net(x)
        x = torch.flatten(x, start_dim=1)
        x= self.fc_net(x)
        return x
    


