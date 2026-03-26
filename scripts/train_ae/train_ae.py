from ayt.utils import save_images, viz_images, create_dir_if_empty, get_module, init_wandb
from ayt.constants import CONFIG_ROOT, RESULT_ROOT
from ayt.autoencoders import AutoEncoder
from ayt.datasets import get_dataset
from ayt.solvers import get_solver
from ayt.tdists import get_tdist
from ayt import torch_utils

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from pytorch_fid.fid_score import calculate_fid_given_paths
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
from torch.func import jvp
from tqdm import tqdm

import numpy as np
import tempfile
import warnings
import logging
import random
import pickle
import hydra
import torch
import wandb
import time
import copy
import sys
import os

sys.modules['torch_utils'] = torch_utils
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def update_ema(net, net_ema, ema_decay):
    for p, p_ema in zip(get_module(net).parameters(),get_module(net_ema).parameters()):
        p_ema.data.copy_(ema_decay * p_ema + (1 - ema_decay) * p)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_dist_reproducibility(base_seed):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    set_seed(base_seed+rank)
    return rank, local_rank

@hydra.main(version_base=None, config_path=CONFIG_ROOT, config_name='train_dae')
def main(cfg: DictConfig):
    
    # Init DDP
    rank, local_rank = setup_dist_reproducibility(base_seed=0)
    device = torch.device("cuda", local_rank)
    
    # Experiment settings
    exp_name = cfg['exp']['name']
    use_wandb = cfg['exp']['use_wandb']
    n_train_iter = cfg['exp']['n_train_iter']

    n_viz = cfg['exp']['n_viz']
    eval_iter = cfg['exp']['eval_every']
    save_iter = cfg['exp']['save_every']

    ckpt_dir = os.path.join(RESULT_ROOT, exp_name, 'ckpts')
    plot_dir = os.path.join(RESULT_ROOT, exp_name, 'plots')
    
    # Dataset and FID stats
    dataset = cfg['dataset']['name']
    bs = cfg['dataset']['batch_size']

    # Dataloader, Solver, UNet, Optimizer
    dataset = get_dataset(cfg['dataset'])
    sampler = DistributedSampler(dataset)
    train_loader = DataLoader(dataset, batch_size=bs, sampler=sampler, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
    
    ae = AutoEncoder().cuda()
    ae_ema = copy.deepcopy(ae)
    ae = DDP(ae, device_ids=[local_rank])
    opt = torch.optim.RAdam(params=ae.parameters(),lr=cfg['optim']['lr'],weight_decay=cfg['optim']['weight_decay'])
    crit = lambda x,y : (x-y).square().mean() + (x-y).abs().mean()

    if rank == 0:
        # Initialize step, iteration, checkpoint directories, wandb
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info('Checkpoints will be saved at [{}]'.format(ckpt_dir))
        create_dir_if_empty(ckpt_dir)
        create_dir_if_empty(plot_dir)
        with open(os.path.join(ckpt_dir,'config.txt'), 'a') as f:
            f.write(OmegaConf.to_yaml(cfg))
        
        if use_wandb:
            init_wandb(cfg)
    
    # Training loop
    s = time.time()
    resume = True
    epoch = iteration = 0
    while resume:
        epoch += 1
        for (x0,y0) in train_loader:
            iteration += 1

            # Sample data and noise
            ae.train()
            x0 = x0.cuda()
            x0_rec = ae(x0)[1]
            
            opt.zero_grad()
            loss = crit(x0,x0_rec)
            loss.backward()
            opt.step()
            
            update_ema(net=ae.module, net_ema=ae_ema, ema_decay=cfg['exp']['ema_decay'])

            e = time.time()
            dur = (e - s) / 3600
            eta = (e - s) / iteration * n_train_iter / 3600

            if rank == 0:
                # Evaluate fid score and visualize images
                if iteration % eval_iter == 0:
                    logger.info('Visualizing Samples')
                    viz_images(x0[:n_viz], os.path.join(plot_dir,'samples.jpg'))
                    viz_images(x0_rec[:n_viz], os.path.join(plot_dir,'samples_recon.jpg'))

                # Logging
                if use_wandb:
                    stats = {'Loss AE' : loss.item()}
                    wandb.log(stats)
                logger.info('[{}] Iteration {} Loss {:.3f} ETA {:.3f}/{:.3f} (hours)'.format(exp_name, iteration, loss.item(), dur, eta))

                # Saving checkpoint
                if (iteration % save_iter == 0):
                    logger.info('Saving model at [{}]'.format(ckpt_dir))
                    torch.save(ae.module.state_dict(), os.path.join(ckpt_dir,'unet_iter{}.pt'.format(iteration)))
                    torch.save(ae_ema.state_dict(), os.path.join(ckpt_dir,'unet_ema_iter{}.pt'.format(iteration)))
        
            # Termination
            if iteration==n_train_iter:
                resume = False
                break
                
            dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()