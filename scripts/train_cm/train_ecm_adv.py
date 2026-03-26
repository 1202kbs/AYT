from ayt.utils import save_images, viz_images, create_dir_if_empty, get_module, init_wandb
from ayt.constants import CONFIG_ROOT, RESULT_ROOT
from ayt.classifiers import get_classifier
from ayt.datasets import get_dataset
from ayt.solvers import get_solver
from ayt.tdists import get_tdist
from ayt.losses import get_loss
from ayt.unets import get_unet
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

@torch.no_grad()
def calculate_fid(fid_stat_dir, unet, solver, FID_bs, noise_loader, image_loader, sigma, aug_lab):
    calculate_dfid = (image_loader is not None)
    
    unet.eval()
    samples = []
    if calculate_dfid:
        for image,noise in tqdm(zip(image_loader,noise_loader)):
            image = image[0][:noise.shape[0]]
            samples.append(solver(image.cuda()+sigma*noise.cuda(),sigma,unet,n_steps=1,augment_labels=aug_lab))
    else:
        for noise in tqdm(noise_loader):
            samples.append(solver(sigma*noise.cuda(),sigma,unet,n_steps=1,augment_labels=aug_lab))
    
    samples = torch.cat(samples,dim=0)
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_images(samples, tmpdirname, start_idx=0)
        fid_score = calculate_fid_given_paths(paths=[tmpdirname,fid_stat_dir], batch_size=FID_bs, device='cuda:0', dims=2048)
    return fid_score, samples

def compute_r1_loss(discriminator, real_data, gamma):
    real_data.requires_grad_(True)
    real_preds = discriminator(real_data)
    gradients = torch.autograd.grad(outputs=real_preds.sum(),inputs=real_data,create_graph=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    grad_norm = gradients.norm(2, dim=1)
    r1_penalty = gamma * grad_norm.pow(2).mean()
    return r1_penalty

def get_noise_dataloader(img_shape, batch_size, num_samples):
    noise = torch.randn((num_samples,) + img_shape)
    return DataLoader(noise, batch_size=batch_size, shuffle=False, drop_last=False)

def filter_state_dict(state_dict):
    res = []
    for n,p in state_dict.items():
        if ('map_augment' not in n):
            res.append((n,p))
    return dict(res)

def update_ema(net, net_ema, ema_decay):
    for p, p_ema in zip(get_module(net).parameters(),get_module(net_ema).parameters()):
        p_ema.data.copy_(ema_decay * p_ema + (1 - ema_decay) * p)

def normalize(a):
    shape = a.shape
    a = a.flatten(start_dim=1)
    a_norm = a.norm(dim=1,keepdim=True).clamp(min=1e-5)
    return (a/a_norm).reshape(shape)


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

@hydra.main(version_base=None, config_path=CONFIG_ROOT, config_name='train_ecm')
def main(cfg: DictConfig):
    
    # Init DDP
    rank, local_rank = setup_dist_reproducibility(base_seed=0)
    device = torch.device("cuda", local_rank)
    
    # Experiment settings
    exp_name = cfg['exp']['name']
    use_wandb = cfg['exp']['use_wandb']
    n_train_iter = cfg['exp']['n_train_iter']

    n_viz = cfg['exp']['n_viz']
    n_FID = cfg['exp']['n_FID']
    FID_bs = cfg['exp']['FID_bs']
    eval_bs = cfg['exp']['eval_bs']
    eval_iter = cfg['exp']['eval_every']
    save_iter = cfg['exp']['save_every']

    ckpt_dir = os.path.join(RESULT_ROOT, exp_name, 'ckpts')
    plot_dir = os.path.join(RESULT_ROOT, exp_name, 'plots')
    
    # Dataset and FID stats
    dataset = cfg['dataset']['name']
    img_size = cfg['dataset']['size']
    img_channels = cfg['dataset']['channels']
    img_shape = (img_channels, img_size, img_size)
    bs = cfg['dataset']['batch_size']
    fid_stat_dir = os.path.join('FID_stats', '{0}-{1}x{1}.npz'.format(dataset,img_size))

    # Dataloader, Solver, UNet, Optimizer
    dataset = get_dataset(cfg['dataset'])
    sampler = DistributedSampler(dataset)
    train_loader = DataLoader(dataset, batch_size=bs, sampler=sampler, num_workers=4, pin_memory=True, worker_init_fn=seed_worker)
    noise_loader = get_noise_dataloader(img_shape, eval_bs, n_FID)
    
    if rank == 0:
        logger.info('Loading unet checkpoint from [{}]'.format(cfg['unet']['ckpt_path']))
    with open(cfg['unet']['ckpt_path'] , 'rb') as f:
        denoiser = pickle.load(f)['ema'].cuda()
    
    unet = get_unet(cfg['unet']).cuda()
    unet.load_state_dict(filter_state_dict(denoiser.state_dict()),strict=False)
    unet_ema = copy.deepcopy(unet)
    unet = DDP(unet, device_ids=[local_rank])
    opt = torch.optim.RAdam(params=unet.parameters(),lr=cfg['optim']['lr'],weight_decay=cfg['optim']['weight_decay'])
    
    clf = get_classifier(cfg['classifier']).cuda()
    # clf = DDP(clf, device_ids=[local_rank]) # Use SyncBatchNorm instead of BatchNorm for DDP!
    opt_clf = torch.optim.RAdam(params=clf.parameters(),lr=cfg['optim']['lr'],weight_decay=cfg['optim']['weight_decay'])
    
    tdist = get_tdist(cfg['tdist'])
    solver = get_solver(cfg['solver'])
    criterion = get_loss(cfg['loss']).cuda()

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
    curr_fid = best_fid = 500
    while resume:
        epoch += 1
        for (x0,y0) in train_loader:
            iteration += 1
            unet.train()
            clf.train()

            # Sample data and noise
            x0, eps = x0.cuda(), torch.randn_like(x0).cuda()
            si, sip1 = tdist.sample(x0.shape[0],iteration)
            xsi, xsip1 = x0 + eps * si.reshape(-1,1,1,1), x0 + eps * sip1.reshape(-1,1,1,1)

            # Consistency Training
            for p in clf.parameters():
                p.requires_grad_(False)
            
            rng_state = torch.cuda.get_rng_state()
            with torch.no_grad():
                D_xsi = unet(xsi,si,rng_state=rng_state)
                D_xsi[si<=0.0] = x0[si<=0.0]
            D_xsip1 = unet(xsip1,sip1,rng_state=rng_state)
            
            w_icm = 1/(sip1-si)
            loss_cm, loss_cm_gnorm = criterion(D_xsip1,D_xsi,w=w_icm)
            
            D_xsip1_cp = D_xsip1.detach().clone().requires_grad_(True)
            loss_adv = (0.0 - clf(D_xsip1_cp)).square().mean()
            loss_adv.backward()
            loss_adv_grad = D_xsip1_cp.grad.detach()
            loss_adv_prox = 0.5 * (D_xsip1 - (D_xsip1-loss_adv_grad).detach()).square().sum(dim=(1,2,3)).mean()
            loss_adv_gnorm = loss_adv_grad.flatten(start_dim=1).norm(dim=1).mean()
            lmda_adv = (0.1 * loss_cm_gnorm / (loss_adv_gnorm + 1e-5)).detach()
            
            opt.zero_grad()
            loss = loss_cm + lmda_adv * loss_adv_prox
            loss.backward()
            opt.step()
            
            update_ema(net=unet.module, net_ema=unet_ema, ema_decay=cfg['exp']['ema_decay'])
            
            # Classifier Training
            for p in clf.parameters():
                p.requires_grad_(True)
            
            opt_clf.zero_grad()
            x_real, x_fake = x0.detach().clone(), D_xsip1.detach().clone()
            loss_clf = (0.0 - clf(x_real)).square().mean() + (1.0 - clf(x_fake)).square().mean()
            loss_clf += compute_r1_loss(clf, x_real, gamma=10.0)
            loss_clf.backward()
            opt_clf.step()

            e = time.time()
            dur = (e - s) / 3600
            eta = (e - s) / iteration * n_train_iter / 3600

            if rank == 0:
                # Evaluate fid score and visualize images
                if iteration % eval_iter == 0:
                    logger.info('Computing FID')
                    curr_fid, samples = calculate_fid(fid_stat_dir, unet_ema, solver, FID_bs, noise_loader, None, get_module(unet).sigma_max, 0.0)
                    viz_images(samples[:n_viz], os.path.join(plot_dir,'samples.jpg'))

                # Logging
                if use_wandb:
                    stats = {'Loss CM'  : loss_cm.item(),
                             'Loss Adv' : loss_adv_prox.item(),
                             'FID'      : curr_fid}
                    wandb.log(stats)
                logger.info('[{}] Iteration {} Loss {:.3f} FID {:.3f} ETA {:.3f}/{:.3f} (hours)'.format(exp_name, iteration, loss.item(), curr_fid, dur, eta))

                # Saving checkpoint
                if (iteration % save_iter == 0):
                    logger.info('Saving model at [{}]'.format(ckpt_dir))
                    torch.save(unet.module.state_dict(), os.path.join(ckpt_dir,'unet_iter{}.pt'.format(iteration)))
                    torch.save(unet_ema.state_dict(), os.path.join(ckpt_dir,'unet_ema_iter{}.pt'.format(iteration)))
                
                # Saving best model checkpoint
                if (iteration % eval_iter == 0) and (curr_fid < best_fid):
                    logger.info('Saving best model at [{}]'.format(ckpt_dir))
                    torch.save(unet.module.state_dict(), os.path.join(ckpt_dir,'unet_best.pt'))
                    torch.save(unet_ema.state_dict(), os.path.join(ckpt_dir,'unet_ema_best.pt'))
                    best_fid = curr_fid
        
            # Termination
            if iteration==n_train_iter:
                resume = False
                break
                
            dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()