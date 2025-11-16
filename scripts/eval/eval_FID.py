from ayt.utils import set_reproducibility, save_images, create_dir_if_empty, get_module
from pytorch_fid.fid_score import calculate_fid_given_paths
from ayt.constants import CONFIG_ROOT, RESULT_ROOT
from ayt.solvers import Euler, EulerMaruyama
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from ayt.datasets import get_dataset
from ayt.unets import get_unet
from tqdm import tqdm

import tempfile
import hydra
import torch
import os

import warnings
warnings.filterwarnings("ignore")

def get_dataloader(img_shape, batch_size, num_samples):
    noise = torch.randn((num_samples,) + img_shape)
    return DataLoader(noise, batch_size=batch_size, shuffle=False, drop_last=False)

@hydra.main(version_base=None, config_path=CONFIG_ROOT, config_name='eval_FID')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Evaluation setting
    exp_name = cfg['exp']['name']
    use_dfid = cfg['exp']['use_dfid']
    n_runs = cfg['exp']['n_runs']
    n_FID = cfg['exp']['n_FID']
    FID_bs = cfg['exp']['FID_bs']
    n_steps = cfg['exp']['n_steps']
    ckpt_dir = cfg['exp']['ckpt_dir']
    s_idx = cfg['exp']['s_idx']
    e_idx = cfg['exp']['e_idx']
    unit = cfg['exp']['unit']
    device_ids = list(range(torch.cuda.device_count()))

    # result_dir = os.path.join(RESULT_ROOT, exp_name)
    # create_dir_if_empty(result_dir)
    result_dir = ckpt_dir

    # Dataset
    dataset = cfg['dataset']['name']
    img_size = cfg['dataset']['size']
    img_channels = cfg['dataset']['channels']
    img_shape = (img_channels, img_size, img_size)
    eval_bs = cfg['dataset']['batch_size']
    fid_stat_dir = os.path.join('FID_stats', '{0}-{1}x{1}.npz'.format(dataset,img_size))

    # Solver
    if n_steps == 1:
        solver = Euler()
        if use_dfid:
            disc = [0.002,0.8]
        else:
            disc = [0.002,80.0]
    elif n_steps == 2:
        solver = EulerMaruyama()
        disc = [0.002,0.821,80.0]

    # Evaluation loop
    FIDs = {}
    ckpt_paths = [os.path.join(ckpt_dir,'unet_ema_iter{}.pt'.format(unit*i)) for i in range(s_idx,e_idx+1)]
    for ckpt_path in tqdm(ckpt_paths):
        print('Evaluating [{}] with discretization {}'.format(ckpt_path,disc))
        FIDs[ckpt_path] = []
        
        # Load UNet
        unet = get_unet(cfg['unet']).cuda()
        unet.load_state_dict(torch.load(ckpt_path, map_location='cuda:0'))
        if len(device_ids) > 1:
            unet = torch.nn.DataParallel(unet, device_ids=device_ids)
        unet.eval()
        
        set_reproducibility(0)
        for _ in range(n_runs):
            dataloader = get_dataloader(img_shape=img_shape, batch_size=eval_bs, num_samples=n_FID)

            samples = []
            if use_dfid:
                image_loader = DataLoader(get_dataset(cfg['dataset']), batch_size=eval_bs, shuffle=False, drop_last=True)
                for image,noise in tqdm(zip(image_loader,dataloader)):
                    sigma_max = disc[-1]
                    samples.append(solver(image[0].cuda()+sigma_max*noise.cuda(), sigma_max, unet, n_steps=n_steps, manual_disc=disc, augment_labels=0.0))
            else:
                for noise in tqdm(dataloader):
                    sigma_max = get_module(unet).sigma_max
                    samples.append(solver(sigma_max*noise.cuda(), sigma_max, unet, n_steps=n_steps, manual_disc=disc, augment_labels=0.0))
            samples = torch.cat(samples,dim=0)[:n_FID]
            
            with tempfile.TemporaryDirectory() as tmpdirname:
                save_images(samples, tmpdirname, start_idx=0)
                FIDs[ckpt_path].append(calculate_fid_given_paths(paths=[tmpdirname,fid_stat_dir], batch_size=FID_bs, device='cuda:0', dims=2048))
            print('FID : {:.3f}'.format(FIDs[ckpt_path][-1]))
    
        torch.save(FIDs, os.path.join(result_dir,'FIDs.pt'))

if __name__ == "__main__":
    main()