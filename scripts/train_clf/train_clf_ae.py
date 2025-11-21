from ayt.utils import set_reproducibility, create_dir_if_empty, get_module, init_wandb
from ayt.constants import CONFIG_ROOT, RESULT_ROOT
from ayt.augmentations import get_augmentation
from ayt.classifiers import get_classifier
from ayt.datasets import get_dataset
from ayt.unets import get_unet
from ayt import torch_utils

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import numpy as np
import warnings
import pickle
import hydra
import torch
import wandb
import time
import sys
import os

sys.modules['torch_utils'] = torch_utils
warnings.filterwarnings("ignore")

def filter_state_dict(state_dict):
    res = []
    for n,p in state_dict.items():
        if ('map_augment' not in n):
            res.append((n,p))
    return dict(res)

@hydra.main(version_base=None, config_path=CONFIG_ROOT, config_name='train_clf')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Experiment settings
    exp_name = cfg['exp']['name']
    use_wandb = cfg['exp']['use_wandb']
    n_train_iter = cfg['exp']['n_train_iter']
    n_rampup_iter = cfg['exp']['n_rampup_iter']
    save_iter = cfg['exp']['save_every']
    bs = cfg['dataset']['batch_size']

    device_ids = list(range(torch.cuda.device_count()))
    ckpt_dir = os.path.join(RESULT_ROOT, exp_name, 'ckpts')
    plot_dir = os.path.join(RESULT_ROOT, exp_name, 'plots')

    # Dataloader, Solver, UNet, Optimizer
    set_reproducibility(0)
    train_loader = DataLoader(get_dataset(cfg['dataset']), batch_size=bs, shuffle=True, drop_last=True)
    clf = get_classifier(cfg['classifier']).cuda()
    
    if cfg['classifier']['ckpt_path'] is not None:
        # For fine-tuning classifier
        clf.load_state_dict(torch.load(cfg['classifier']['ckpt_path'],weights_only=True),strict=False)
        print('Loading classifier checkpoint from [{}]'.format(cfg['classifier']['ckpt_path']))
    else:
        print('No checkpoint provided for classifier. Using random initialization.')
    
    if len(device_ids) > 1:
        clf = torch.nn.DataParallel(clf, device_ids=device_ids)
    opt_clf = torch.optim.RAdam(params=clf.parameters(),lr=1e-4)
    
    aug = get_augmentation(cfg['aug'])
    
    unet = get_unet(cfg['unet']).cuda()
    if cfg['unet']['ckpt_path'] is not None:
        if 'pkl' in cfg['unet']['ckpt_path']:
            with open(cfg['unet']['ckpt_path'] , 'rb') as f:
                ckpt = pickle.load(f)['ema'].state_dict()
                unet.load_state_dict(filter_state_dict(ckpt),strict=False)
                unet.eval()
    
    # Initialize step, iteration, checkpoint directories, wandb
    assert cfg['classifier']['n_classes'] == (cfg['aug']['n_labs']+1), "Classifier output dimension must equal number of augmentation labels."
    print('Checkpoints will be saved at [{}]'.format(ckpt_dir))
    create_dir_if_empty(ckpt_dir)
    create_dir_if_empty(plot_dir)
    with open(os.path.join(ckpt_dir,'config.txt'), 'a') as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    if use_wandb:
        init_wandb(cfg)
    
    # Training loop
    s = time.time()
    iteration = 0
    nl = 1e-3 * torch.ones(size=[cfg['aug']['n_labs']+1]).cuda()
    while True:
        for (x0,y0) in train_loader:
            
            # Sample data and noise
            ru_factor = min(1.0,(iteration+1)/n_rampup_iter)
            aug.p = ru_factor * cfg['aug']['p']
            
            x_aug = x0.cuda()
            ae_w = torch.rand([bs]).cuda() < aug.p
            sigma = (0.5 * torch.rand(size=[bs]).cuda()).clamp(min=0.002)
            with torch.no_grad():
                if ae_w.sum() > 0:
                    x_rec = unet((x_aug+sigma.reshape(-1,1,1,1)*torch.randn_like(x_aug))[ae_w],sigma[ae_w])
                    x_aug[ae_w] = x_rec
                    ae_w = ae_w.float()
            
            x_aug, labs = aug(x_aug,torch.ones(size=[bs]).cuda())
            labs = torch.cat([ae_w.reshape(-1,1),labs],dim=1)
            
            # update normalizing factor
            labs_max = labs.abs().max(dim=0)[0]
            nl = nl.clamp(min=labs_max)

            clf.train()
            opt_clf.zero_grad()
            labs_pred = clf(x_aug)
            loss_clf = (labs/nl.reshape(1,-1)-labs_pred).square().mean()
            loss_clf_ae_abs = (labs[:,0]-nl[0]*labs_pred[:,0])[ae_w.flatten()>0].abs().mean()
            loss_clf_abs = (labs-nl.reshape(1,-1)*labs_pred).abs().mean()
            loss_clf.backward()
            opt_clf.step()

            iteration += 1
            e = time.time()
            dur = (e - s) / 3600
            eta = (e - s) / iteration * n_train_iter / 3600

            # Logging
            if use_wandb:
                stats = {'Loss CLF'    : loss_clf.item(),
                         'Loss CLF ABS' : loss_clf_abs.item(),
                         'Loss CLF AE ABS' : loss_clf_ae_abs.item()}
                wandb.log(stats)
            print('[{}] Iteration {} Loss {:.3f} ETA {:.3f}/{:.3f} (hours)'.format(exp_name, iteration, loss_clf.item(), dur, eta))

            # Saving checkpoint
            if (iteration % save_iter == 0):
                print('Saving model at [{}]'.format(ckpt_dir))
                torch.save(get_module(clf).state_dict(), os.path.join(ckpt_dir,'clf_iter{}.pt'.format(iteration)))
        
            # Termination
            if iteration==n_train_iter:
                return

if __name__ == "__main__":
    main()