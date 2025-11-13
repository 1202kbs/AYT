from ayt.utils import set_reproducibility, save_images, viz_images, create_dir_if_empty, get_module, init_wandb
from ayt.constants import CONFIG_ROOT, RESULT_ROOT
from ayt.datasets import get_dataset
from ayt.solvers import get_solver
from ayt.tdists import get_tdist
from ayt.losses import get_loss
from ayt.unets import get_unet
from ayt import torch_utils

from pytorch_fid.fid_score import calculate_fid_given_paths
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import tempfile
import warnings
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

@hydra.main(version_base=None, config_path=CONFIG_ROOT, config_name='train_ecm')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

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

    device_ids = list(range(torch.cuda.device_count()))
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
    set_reproducibility(0)
    noise_loader = get_noise_dataloader(img_shape, eval_bs, n_FID)
    image_loader = DataLoader(get_dataset(cfg['dataset']), batch_size=eval_bs)
    train_loader = DataLoader(get_dataset(cfg['dataset']), batch_size=bs, shuffle=True, drop_last=True)
    unet = get_unet(cfg['unet']).cuda()

    if cfg['unet']['ckpt_path'] is not None:
        with open(cfg['unet']['ckpt_path'] , 'rb') as f:
            ckpt = pickle.load(f)['ema'].state_dict()
            unet.load_state_dict(filter_state_dict(ckpt),strict=False)
        print('Loading unet checkpoint from [{}]'.format(cfg['unet']['ckpt_path']))
    else:
        print('No checkpoint provided for unet. Using random initialization.')

    unet_ema = copy.deepcopy(unet)
    if len(device_ids) > 1:
        unet = torch.nn.DataParallel(unet, device_ids=device_ids)
        unet_ema = torch.nn.DataParallel(unet_ema, device_ids=device_ids)

    opt = torch.optim.RAdam(params=unet.parameters(),
                            lr=cfg['optim']['lr'],
                            weight_decay=cfg['optim']['weight_decay'])
    
    tdist = get_tdist(cfg['tdist'])
    solver = get_solver(cfg['solver'])
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
    curr_fid = best_fid = 500
    curr_dfids = {0.2 : 500, 0.4 : 500, 0.8 : 500}
    while True:
        for (x0,y0) in train_loader:

            # Sample data and noise
            x0, eps = x0.cuda(), torch.randn_like(x0).cuda()
            si, sip1 = tdist.sample(bs,iteration)
            xsi, xsip1 = x0 + eps * si.reshape(-1,1,1,1), x0 + eps * sip1.reshape(-1,1,1,1)

            # Consistency Training Variables
            unet.train()
            rng_state = torch.cuda.get_rng_state()
            with torch.no_grad():
                D_xsi = unet(xsi,si,rng_state=rng_state)
                D_xsi[si<=0.0] = x0[si<=0.0]

            D_xsip1 = unet(xsip1,sip1,rng_state=rng_state)

            # Optimize UNet
            w_icm = 1/(sip1-si)
            if cfg['loss']['name'] == 'pseudohuber':
                loss_cm = criterion(D_xsip1,D_xsi,w=w_icm)[0]
                loss = loss_cm
            else:
                D_xsip1_cp = D_xsip1.detach().clone().requires_grad_(True)
                loss_cm = criterion(D_xsip1_cp,D_xsi,w=w_icm)[0]
                loss_cm.backward()
                loss_cm_grad = normalize(D_xsip1_cp.grad.detach().clone())
                loss_cm_prox = (w_icm * 0.5 * (D_xsip1 - (D_xsip1-loss_cm_grad).detach()).square().sum(dim=(1,2,3))).mean()
                loss = loss_cm_prox
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            update_ema(net=unet, net_ema=unet_ema, ema_decay=cfg['exp']['ema_decay'])

            iteration += 1
            e = time.time()
            dur = (e - s) / 3600
            eta = (e - s) / iteration * n_train_iter / 3600

            # Evaluate fid score and visualize images
            if iteration % eval_iter == 0:
                print('Computing FID')
                curr_fid, samples = calculate_fid(fid_stat_dir, unet_ema, solver, FID_bs, noise_loader, None, get_module(unet).sigma_max, 0.0)
                viz_images(samples[:n_viz], os.path.join(plot_dir,'samples.jpg'))
                for sigma in curr_dfids.keys():
                    print('Computing dFID sigma = {:.1f}'.format(sigma))
                    curr_dfid, _ = calculate_fid(fid_stat_dir, unet_ema, solver, FID_bs, noise_loader, image_loader, sigma, 0.0)
                    curr_dfids[sigma] = curr_dfid

            # Logging
            if use_wandb:
                stats = {'Loss CM' : loss_cm.item(),
                        'FID' : curr_fid}
                stats = stats | {'dFID {:.1f}'.format(k) : v for (k,v) in curr_dfids.items()}
                wandb.log(stats)
            print('[{}] Iteration {} Loss {:.3f} FID {:.3f} ETA {:.3f}/{:.3f} (hours)'.format(exp_name, iteration, loss.item(), curr_fid, dur, eta))

            # Saving checkpoint
            if (iteration % save_iter == 0):
                print('Saving model at [{}]'.format(ckpt_dir))
                torch.save(get_module(unet).state_dict(), os.path.join(ckpt_dir,'unet_iter{}.pt'.format(iteration)))
                torch.save(get_module(unet_ema).state_dict(), os.path.join(ckpt_dir,'unet_ema_iter{}.pt'.format(iteration)))
            
            # Saving best model checkpoint
            if (iteration % eval_iter == 0) and (curr_fid < best_fid):
                print('Saving best model at [{}]'.format(ckpt_dir))
                torch.save(get_module(unet).state_dict(), os.path.join(ckpt_dir,'unet_best.pt'))
                torch.save(get_module(unet_ema).state_dict(), os.path.join(ckpt_dir,'unet_ema_best.pt'))
                best_fid = curr_fid
        
            # Termination
            if iteration==n_train_iter:
                return

if __name__ == "__main__":
    main()