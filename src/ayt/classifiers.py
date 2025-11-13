import torch.nn.functional as F
import torch.nn as nn
import torch

from einops import rearrange
from einops.layers.torch import Rearrange


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

##############################################################################################################
##############################################################################################################
# ViT

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        feats = []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            feats.append(x.clone())
        return self.norm(x), feats

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img, return_feats=False):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x, feats = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        x = self.linear_head(x)
        feats.append(x.clone())
        
        if return_feats:
            return feats
        else:
            return x
    
    def distance(self,x,y,reduce_mean=False):
        diffs = 0
        x_feats = self.forward(x,return_feats=True)
        y_feats = self.forward(y,return_feats=True)
        for i in range(len(x_feats)):
            xf, yf = x_feats[i], y_feats[i]
            if i < len(x_feats)-1:
                diffs += ph(xf-yf,dim=2).sum(dim=1)
            else:
                diffs += ph((xf-yf)[...,None],dim=2).sum(dim=1)
        
        if reduce_mean:
            diffs = diffs.mean()
        return diffs


##############################################################################################################
##############################################################################################################
# ResNet

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feats=False):
        feats = []
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        feats.append(out.clone())
        out = self.layer2(out)
        feats.append(out.clone())
        out = self.layer3(out)
        feats.append(out.clone())
        out = self.layer4(out)
        feats.append(out.clone())
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        feats.append(out.clone())
        
        if return_feats:
            return feats
        else:
            return out
    
    def distance(self,x,y,reduce_mean=False):
        diffs = 0
        x_feats = self.forward(x,return_feats=True)
        y_feats = self.forward(y,return_feats=True)
        for i in range(len(x_feats)):
            xf, yf = x_feats[i], y_feats[i]
            if i < len(x_feats)-1:
                diffs += ph(xf-yf,dim=(2,3)).sum(dim=1)
            else:
                diffs += ph((xf-yf)[...,None],dim=2).sum(dim=1)
        
        if reduce_mean:
            diffs = diffs.mean()
        return diffs

##############################################################################################################
##############################################################################################################


def get_classifier(cfg):
    if 'VGG' in cfg['name']:
        return VGG(vgg_name=cfg['name'],
                img_resolution=cfg['img_resolution'],
                in_channels=cfg['in_channels'],
                n_classes=cfg['n_classes'])
    elif 'ViT' in cfg['name']:
        return SimpleViT(
                image_size=cfg['img_resolution'],
                patch_size=int(cfg['img_resolution']//8),
                num_classes=cfg['n_classes'],
                dim=cfg['dim'],
                depth=cfg['depth'],
                heads=cfg['heads'],
                mlp_dim=2048)
    elif 'ResNet' in cfg['name']:
        return ResNet(block=BasicBlock,
                      num_blocks=[2,2,2,2],
                      num_classes=cfg['n_classes'])