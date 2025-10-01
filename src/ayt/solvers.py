from ayt.utils import get_module

import torch

def get_solver(cfg):
    if cfg['name'] == 'euler':
        return Euler()
    elif cfg['name'] == 'eulermaruyama':
        return EulerMaruyama()

def get_edm_schedule(n_steps,sigma,sigma_min,sigma_max):
    sigma = max(min(sigma,sigma_max),sigma_min)
    if n_steps==1:
        # disc = torch.tensor([0.0,sigma])
        disc = torch.tensor([sigma_min,sigma])
    else:
        disc = torch.linspace(sigma_min**(1/7), sigma**(1/7), n_steps)**7
        disc = torch.cat([torch.tensor([0.0]),disc])
    return disc

class Euler:

    @torch.no_grad()
    def __call__(self, x, sigma, net, n_steps, class_labels=None, augment_labels=0.0, manual_disc=None):
        # Create sigma discretization
        if manual_disc is not None:
            n_steps = len(manual_disc)-1
            disc = torch.tensor(sorted(manual_disc)).flip(0)
        else:
            sigma_min = net.module.sigma_min if type(net)==torch.nn.DataParallel else net.sigma_min
            sigma_max = net.module.sigma_max if type(net)==torch.nn.DataParallel else net.sigma_max
            disc = get_edm_schedule(n_steps,sigma,sigma_min,sigma_max).flip(0)

        # Create random class labels if network is conditional and no class labels are given
        label_dim, augment_dim = get_module(net).label_dim, get_module(net).augment_dim
        augment_labels = augment_labels*torch.ones([x.shape[0],augment_dim]).to(x.device)
        if class_labels is None and label_dim > 0:
            class_labels = torch.eye(label_dim)[torch.randint(label_dim, size=[x.shape[0]])].to(x.device)
        
        # Generation
        for i in range(n_steps):
            D_x = net(x,disc[i]*torch.ones(size=[x.shape[0]]).to(x.device),class_labels=class_labels,augment_labels=augment_labels)
            V_x = (x - D_x) / disc[i]
            d_sigma = disc[i+1] - disc[i]
            x = x + V_x * d_sigma
        
        return x

class Heun:
    
    @torch.no_grad()
    def __call__(self, x, sigma, net, n_steps, class_labels=None, augment_labels=0.0, manual_disc=None):
        # Create sigma discretization
        if manual_disc is not None:
            n_steps = len(manual_disc)-1
            disc = torch.tensor(sorted(manual_disc)).flip(0)
        else:
            sigma_min = net.module.sigma_min if type(net)==torch.nn.DataParallel else net.sigma_min
            sigma_max = net.module.sigma_max if type(net)==torch.nn.DataParallel else net.sigma_max
            disc = get_edm_schedule(n_steps,sigma,sigma_min,sigma_max).flip(0)

        # Create random class labels if network is conditional and no class labels are given
        label_dim, augment_dim = get_module(net).label_dim, get_module(net).augment_dim
        augment_labels = augment_labels*torch.ones([x.shape[0],augment_dim]).to(x.device)
        if class_labels is None and label_dim > 0:
            class_labels = torch.eye(label_dim)[torch.randint(label_dim, size=[x.shape[0]])].to(x.device)

        for i in range(n_steps):
            D_x = net(x,disc[i]*torch.ones(size=[x.shape[0]]).to(x.device),class_labels=class_labels,augment_labels=augment_labels)
            V_x = (x - D_x) / disc[i]
            d_sigma = disc[i+1] - disc[i]
            x_next = x + V_x * d_sigma
            if i < (n_steps-1):
                D_x_next = net(x_next,disc[i+1]*torch.ones(size=[x.shape[0]]).to(x.device),class_labels=class_labels,augment_labels=augment_labels)
                V_x_next = (x_next - D_x_next) / disc[i+1]
                x = x + 0.5 * (V_x + V_x_next) * d_sigma
            else:
                x = x_next
        return x

class EulerMaruyama:

    @torch.no_grad()
    def __call__(self, x, sigma, net, n_steps, class_labels=None, augment_labels=0.0, manual_disc=None):
        # Create sigma discretization
        if manual_disc is not None:
            n_steps = len(manual_disc)-1
            disc = torch.tensor(sorted(manual_disc)).flip(0)
        else:
            sigma_min = net.module.sigma_min if type(net)==torch.nn.DataParallel else net.sigma_min
            sigma_max = net.module.sigma_max if type(net)==torch.nn.DataParallel else net.sigma_max
            disc = get_edm_schedule(n_steps,sigma,sigma_min,sigma_max).flip(0)

        # Create random class labels if network is conditional and no class labels are given
        label_dim, augment_dim = get_module(net).label_dim, get_module(net).augment_dim
        augment_labels = augment_labels*torch.ones(x.shape[0],augment_dim).to(x.device) if augment_dim > 0 else None
        if class_labels is None and label_dim > 0:
            class_labels = torch.eye(label_dim)[torch.randint(label_dim, size=[x.shape[0]])].to(x.device)
        
        # Generation
        for i in range(n_steps):
            if i < n_steps-1:
                # x = net(x,disc[i]*torch.ones(size=[x.shape[0]]).to(x.device),class_labels=class_labels,augment_labels=augment_labels)
                # x = x + disc[i+1] * torch.randn_like(x)

                D_x = net(x,disc[i]*torch.ones(size=[x.shape[0]]).to(x.device),class_labels=class_labels,augment_labels=augment_labels)
                V_x = (x - D_x) / disc[i]
                d_sigma = 0.002 - disc[i]
                x = x + V_x * d_sigma
                x = x + (disc[i+1].square()-0.002**2).sqrt() * torch.randn_like(x)
            else:
                D_x = net(x,disc[i]*torch.ones(size=[x.shape[0]]).to(x.device),class_labels=class_labels,augment_labels=augment_labels)
                V_x = (x - D_x) / disc[i]
                d_sigma = disc[i+1] - disc[i]
                x = x + V_x * d_sigma
        return x