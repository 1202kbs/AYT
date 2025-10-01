import numpy as np
import torch
import lpips

def get_loss(cfg):
    if cfg['name'] == 'l1':
        return L1()
    elif cfg['name'] == 'l2':
        return L2()
    elif cfg['name'] == 'pseudohuber':
        return PseudoHuber(c_mult=cfg['c_mult'])
    elif cfg['name'] == 'lpips':
        return LPIPS()

class L1(torch.nn.Module):
    def __call__(self,x,y,w=None):
        dim = x[0].numel()
        diff = (x-y).flatten(start_dim=1).abs().sum(dim=1)
        diff_gnorm = dim*torch.ones_like(diff)
        w = torch.ones(size=[x.shape[0]]).to(x.device) if w is None else w.flatten()
        return (w*diff).mean(), (w*diff_gnorm).mean()

class L2(torch.nn.Module):
    def __call__(self,x,y,w=None):
        diff = (x-y).flatten(start_dim=1).square().sum(dim=1)
        diff_gnorm = 2*diff.sqrt()
        w = torch.ones(size=[x.shape[0]]).to(x.device) if w is None else w.flatten()
        return (w*diff).mean(), (w*diff_gnorm).mean()

class PseudoHuber(torch.nn.Module):
    def __init__(self,c_mult=0.00054):
        super().__init__()
        self.c_mult = c_mult
    
    def __call__(self,x,y,w=None):
        dim = x[0].numel()
        c = self.c_mult * np.sqrt(dim)
        diff = (x-y).flatten(start_dim=1).square().sum(dim=1)
        diff = (diff+c**2).sqrt()-c
        diff_gnorm = diff.sqrt()/(diff+c**2).sqrt()
        w = torch.ones(size=[x.shape[0]]).to(x.device) if w is None else w.flatten()
        return (w*diff).mean(), (w*diff_gnorm).mean()

class LPIPS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips_loss = lpips.LPIPS(net='vgg')
    
    def __call__(self,x,y,w=None):
        w = torch.ones(size=[x.shape[0]]).to(x.device) if w is None else w.flatten()
        return (w*self.lpips_loss(x,y).flatten()).mean(), 0.0