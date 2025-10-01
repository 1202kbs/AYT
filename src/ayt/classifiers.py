import torch.nn as nn
import torch

##############################################################################################################
##############################################################################################################
# VGG

vgg_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def ph(x,dim,eps=1e-5):
    return (x.square().sum(dim=dim)+eps**2).sqrt()-eps

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class VGG(nn.Module):
    def __init__(self, vgg_name, img_resolution, in_channels, n_classes):
        super(VGG, self).__init__()
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.n_classes = n_classes

        # Make VGG layers
        cfg = vgg_cfg[vgg_name]
        res, nc, block_idx = self.img_resolution, self.in_channels, 0
        self.layers = torch.nn.ModuleDict()
        pool, block = (nn.MaxPool2d, VGGBlock)
        for x in cfg:
            if x == 'M':
                self.layers[f'{res}x{res}_maxpool'] = pool(kernel_size=2,stride=2)
                res = res >> 1
                block_idx = 0
            else:
                self.layers[f'{res}x{res}_block{block_idx}'] = block(nc,x)
                block_idx += 1
                nc = x
        self.layers[f'{res}x{res}_avgpool'] = nn.AvgPool2d(kernel_size=res,stride=res)
        self.layers[f'classifier'] = nn.Linear(512, self.n_classes)

    def forward(self, x, return_feats=False):
        feats = []
        for name, layer in self.layers.items():
            if 'classifier' in name:
                x = x.flatten(start_dim=1)
                x = layer(x)
                feats.append(x[...,None,None].clone())
            elif 'maxpool' in name:
                x = layer(x)
                feats.append(x.clone())
            else:
                x = layer(x)
        
        if return_feats:
            return feats
        else:
            return x

    def __get_diff__(self,x,y,w=None):
        '''
        x,y : tensors of shape NxCxHxW
        w : weight tensor of shape NxC
        '''
        return (w*ph(x-y,dim=(2,3))).sum(dim=1)

    def distance(self,x,y,reduce_mean=False):
        diffs = 0
        x_feats = self.forward(x,return_feats=True)
        y_feats = self.forward(y,return_feats=True)
        for i in range(len(x_feats)):
            xf, yf = x_feats[i], y_feats[i]
            w = torch.ones(size=[1,xf.shape[1]]).cuda()
            diffs += self.__get_diff__(xf,yf,w)
        if reduce_mean:
            diffs = diffs.mean()
        return diffs

def get_classifier(cfg):
    return VGG(vgg_name=cfg['name'],
               img_resolution=cfg['img_resolution'],
               in_channels=cfg['in_channels'],
               n_classes=cfg['n_classes'])