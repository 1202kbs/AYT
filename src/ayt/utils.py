import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np

import random
import torch
import wandb
import PIL
import os

def init_wandb(cfg):
    wandb.init(
        project='Your-Project-Name',
        name=cfg['exp']['name'],
        config={
            'lr' : cfg['optim']['lr'],
        },
    )

def get_module(net):
    if type(net) in [torch.nn.DataParallel,torch.nn.parallel.DistributedDataParallel]:
        return net.module
    else:
        return net

def set_reproducibility(seed):
    print("Using seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    cudnn.benchmark = False

def get_rng_states():
    rng_states = []
    rng_states.append(np.random.get_state())
    rng_states.append(torch.get_rng_state())
    rng_states.append(torch.cuda.get_rng_state())
    return rng_states

def set_rng_states(rng_states):
    np.random.set_state(rng_states[0])
    torch.set_rng_state(rng_states[1])
    torch.cuda.set_rng_state(rng_states[2])

def set_requires_grad(net,state):
    for p in net.parameters():
        p.requires_grad = state

def tensor2img(X):
    X = (X * 127.5 + 128).clip(0,255).to(torch.uint8)
    if X.shape[1]==3:
        return X.permute(0,2,3,1).detach().cpu().numpy()
    else:
        return X[:,0][...,None].repeat([1,1,1,3]).detach().cpu().numpy()

def save_images(X, save_dir, start_idx=0):
    X = tensor2img(X)
    for i in range(X.shape[0]):
        img = PIL.Image.fromarray(X[i], 'RGB')
        img.save(os.path.join(save_dir, '{}.png'.format(i+start_idx)))

def viz_images(tensors, path):
    N = int(np.sqrt(tensors.shape[0]))
    imgs = tensor2img(tensors)
    plt.figure(figsize=(1.5*N,1.5*N))
    for i in range(N**2):
        plt.subplot(N,N,i+1)
        plt.imshow(imgs[i])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()

def create_dir_if_empty(path):
    if not os.path.isdir(path):
        os.makedirs(path)