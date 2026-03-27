import torch.nn as nn
import torch

def ph(x,dim,eps=1e-5):
    return (x.square().sum(dim=dim)+eps**2).sqrt()-eps

######################################################################################
# AutoEncoder Building Blocks

class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.gelu = nn.GELU()
        self.gn1 = nn.GroupNorm(32, in_planes) if in_planes > 32 else nn.Identity()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, planes) if planes > 32 else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.gelu(self.gn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.gelu(self.gn2(out)))
        out += shortcut
        return out

class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.gelu = nn.GELU()
        self.gn1 = nn.GroupNorm(32, in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.gn2 = nn.GroupNorm(32, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(32, planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.gelu(self.gn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.gelu(self.gn2(out)))
        out = self.conv3(self.gelu(self.gn3(out)))
        out += shortcut
        return out

class UpsamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpsamplingBlock, self).__init__()
        self.gelu = nn.GELU()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        # self.gn1 = nn.GroupNorm(min(32,in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.gn2 = nn.GroupNorm(min(32,out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        x = self.upsample(x)
        # x = self.conv1(self.gelu(self.gn1(x)))
        # x = self.conv2(self.gelu(self.gn2(x)))
        x = self.conv1(self.gelu(x))
        x = self.conv2(self.gelu(x))
        return x

class ResidualUpsamplingBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super(ResidualUpsamplingBlock, self).__init__()
        self.skip_channels = skip_channels
        self.gelu = nn.GELU()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.gn1 = nn.GroupNorm(min(32,in_channels+skip_channels), in_channels+skip_channels)
        self.conv1 = nn.Conv2d(in_channels+skip_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(32,out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x, x_skip):
        x = self.upsample(x)
        x = torch.cat([x,x_skip],dim=1)
        x = self.conv1(self.gelu(self.gn1(x)))
        x = self.conv2(self.gelu(self.gn2(x)))
        return x

######################################################################################
# AutoEncoders

class Encoder(nn.Module):
    def __init__(self, img_resolution, in_channels, lat_channels, enc_block, num_enc_blocks):
        super(Encoder, self).__init__()
        enc_block = globals()[enc_block]

        res, nc, planes, block_idx = img_resolution, 64, 64, 0
        self.layers = torch.nn.ModuleDict()
        self.layers[f'{res}x{res}_conv'] = nn.Conv2d(in_channels, nc, kernel_size=3, stride=1, padding=1, bias=False)
        for i,num_blocks in enumerate(num_enc_blocks):
            strides = [1 if i ==0 else 2] + [1]*(num_blocks-1)
            res = res if i == 0 else (res >> 1)
            block_idx = 0
            for j in range(num_blocks):
                layer_name = f'{res}x{res}_skip' if j == (num_blocks-1) else f'{res}x{res}_block{block_idx}'
                self.layers[layer_name] = enc_block(in_planes=nc,planes=planes,stride=strides[j])
                nc = planes * enc_block.expansion
                block_idx += 1
            planes = planes << 1
        self.layers[f'{res}x{res}_compress'] = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(nc, lat_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(lat_channels)
        )
    
    def forward(self, x):
        feats = []
        for name, layer in self.layers.items():
            x = layer(x)
            if 'skip' in name:
                feats.append(x)
        return x, feats

class Decoder(nn.Module):
    def __init__(self, in_resolution, in_channels, out_channels, n_upsample):
        super(Decoder, self).__init__()
        self.gelu = nn.GELU()
        self.layers = torch.nn.ModuleDict()
        nc = 32 * 2 ** n_upsample
        res = in_resolution
        
        self.layers[f'block_init'] = PreActBlock(in_channels, nc, stride=1)
        for _ in range(n_upsample):
            res = res << 1
            self.layers[f'{res}x{res}_block'] = UpsamplingBlock(nc, nc >> 1)
            nc = nc >> 1
        self.layers[f'block_final'] = PreActBlock(nc, out_channels, stride=1)
    
    def forward(self, x):
        for name, layer in self.layers.items():
            x = layer(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, img_resolution=32, in_channels=3, lat_channels=32, num_enc_blocks=[2,2,2,2], enc_block='PreActBlock'):
        super(AutoEncoder, self).__init__()
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.lat_channels = lat_channels

        n_down = n_up = len(num_enc_blocks)-1
        lat_res = img_resolution >> n_down
        self.enc = Encoder(img_resolution, in_channels, lat_channels, enc_block, num_enc_blocks)
        self.dec = Decoder(lat_res, lat_channels, in_channels, n_up)
    
    def forward(self, x, return_feats=False):
        z, feats = self.enc(x)
        x = self.dec(z + torch.randn_like(z) * 0.05)
        
        if return_feats:
            return feats
        else:
            return z, x
    
    def distance(self,x,y,reduce_mean=False):
        diffs = 0
        x_feats = self.forward(x,return_feats=True)
        y_feats = self.forward(y,return_feats=True)
        for i in range(len(x_feats)):
            xf, yf = x_feats[i], y_feats[i]
            diffs += ph(xf-yf,dim=(2,3)).sum(dim=1)
        
        if reduce_mean:
            diffs = diffs.mean()
        return diffs

# def get_autoencoder(cfg):
#     if cfg['name'] == 'ResidualAutoEncoder':
#         return ResidualAutoEncoder(img_resolution=cfg['img_resolution'],
#                                    in_channels=cfg['in_channels'],
#                                    enc_block=cfg['enc_block'],
#                                    num_enc_blocks=cfg['num_enc_blocks'])
#     elif cfg['name'] == 'MultiResAutoEncoder':
#         return MultiResAutoEncoder(img_resolution=cfg['img_resolution'],
#                                    in_channels=cfg['in_channels'],
#                                    lat_channels=cfg['lat_channels'],
#                                    enc_block=cfg['enc_block'],
#                                    num_enc_blocks=cfg['num_enc_blocks'])