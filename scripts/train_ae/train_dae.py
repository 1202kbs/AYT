from ayt.utils import set_reproducibility, viz_images, create_dir_if_empty, get_module, init_wandb
from ayt.constants import CONFIG_ROOT, RESULT_ROOT
from ayt.datasets import get_dataset
from ayt.autoencoders import get_ae
from ayt.losses import get_loss
from ayt import torch_utils

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import warnings
import hydra
import torch
import wandb
import time
import sys
import os

sys.modules['torch_utils'] = torch_utils
warnings.filterwarnings("ignore")

@hydra.main(version_base=None, config_path=CONFIG_ROOT, config_name='train_dae')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Experiment settings
    exp_name = cfg['exp']['name']
    use_wandb = cfg['exp']['use_wandb']
    n_train_iter = cfg['exp']['n_train_iter']
    eval_iter = cfg['exp']['eval_every']
    sigma_max = cfg['exp']['sigma_max']

    n_viz = cfg['exp']['n_viz']
    save_iter = cfg['exp']['save_every']

    device_ids = list(range(torch.cuda.device_count()))
    ckpt_dir = os.path.join(RESULT_ROOT, exp_name, 'ckpts')
    plot_dir = os.path.join(RESULT_ROOT, exp_name, 'plots')
    
    # Dataset and FID stats
    bs = cfg['dataset']['batch_size']

    # Dataloader, Solver, UNet, Optimizer
    set_reproducibility(0)
    train_loader = DataLoader(get_dataset(cfg['dataset']), batch_size=bs, shuffle=True, drop_last=True)
    ae = get_ae(cfg['autoencoder']).cuda()
    if len(device_ids) > 1:
        ae = torch.nn.DataParallel(ae, device_ids=device_ids)

    opt = torch.optim.RAdam(params=ae.parameters(),
                            lr=cfg['optim']['lr'],
                            weight_decay=cfg['optim']['weight_decay'])
    
    criterion = get_loss(cfg['loss']).cuda()
    
    # Initialize step, iteration, checkpoint directories, wandb
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
    while True:
        for (x0,y0) in train_loader:

            # Sample data and noise
            x0, eps = x0.cuda(), torch.randn_like(x0).cuda()
            sigma = sigma_max * torch.rand(size=[x0.shape[0],1,1,1]).cuda()
            xs = x0 + sigma * eps

            # Consistency Training Variables
            ae.train()
            x0_pred = ae(xs)
            loss_ae = criterion(x0_pred,x0)[0]
            loss_ae.backward()
            opt.step()
            opt.zero_grad()

            iteration += 1
            e = time.time()
            dur = (e - s) / 3600
            eta = (e - s) / iteration * n_train_iter / 3600

            # Evaluate fid score and visualize images
            if iteration % eval_iter == 0:
                viz_images(x0[:n_viz], os.path.join(plot_dir,'samples.jpg'))
                viz_images(xs[:n_viz], os.path.join(plot_dir,'samples_noise.jpg'))
                viz_images(x0_pred[:n_viz], os.path.join(plot_dir,'samples_recon.jpg'))

            # Logging
            if use_wandb:
                stats = {'Loss AE' : loss_ae.item()}
                wandb.log(stats)
            print('[{}] Iteration {} Loss {:.3f} ETA {:.3f}/{:.3f} (hours)'.format(exp_name, iteration, loss_ae.item(), dur, eta))

            # Saving checkpoint
            if (iteration % save_iter == 0):
                print('Saving model at [{}]'.format(ckpt_dir))
                torch.save(get_module(ae).state_dict(), os.path.join(ckpt_dir,'unet_iter{}.pt'.format(iteration)))
        
            # Termination
            if iteration==n_train_iter:
                return

if __name__ == "__main__":
    main()