from scipy.special import erf

import torch

def get_tdist(cfg):
    if cfg['name'] == 'lognormal':
        return Lognormal(mu=cfg['mu'], std=cfg['std'])
    elif cfg['name'] == 'continuouslognormal':
        return ContinuousLognormal(mu=cfg['mu'], std=cfg['std'], q=cfg['q'], k=cfg['k'],
                                   b=cfg['b'], n_stages=cfg['n_stages'], adj=cfg['adj'])
    elif cfg['name'] == 'uniform':
        return Uniform()

class Uniform:

    def sample(self, num_samples, disc):
        cdf = disc
        pdf = cdf[1:] - cdf[:-1]
        disc_idx = torch.multinomial(pdf, num_samples=num_samples, replacement=True)
        return disc[disc_idx].cuda(), disc[disc_idx+1].cuda()

class Lognormal:

    def __init__(self,mu,std):
        self.mu = mu
        self.std = std

    def sample(self, num_samples, disc):
        cdf = erf((disc.log() - self.mu) / self.std)
        pdf = cdf[1:] - cdf[:-1]
        disc_idx = torch.multinomial(pdf, num_samples=num_samples, replacement=True)
        return disc[disc_idx].cuda(), disc[disc_idx+1].cuda()

class ContinuousLognormal:

    def __init__(self, mu, std, q=2, k=8.0, b=1.0, n_stages=8, n_train_iter=400000, adj='sigmoid'):
        self.mu = mu
        self.std = std
        
        if adj == 'const':
            self.t_to_r = self.t_to_r_const
        elif adj == 'sigmoid':
            self.t_to_r = self.t_to_r_sigmoid
        else:
            raise ValueError(f'Unknow schedule type {adj}!')

        self.q = q
        self.k = k
        self.b = b
        self.n_stages = n_stages
        self.n_train_iter = n_train_iter
    
    def t_to_r_const(self, t, iteration):
        d = self.n_train_iter // self.n_stages
        stage = iteration // d
        decay = max(1 / self.q ** (stage+1),1/256)
        ratio = 1 - decay
        r = t * ratio
        return torch.clamp(r, min=0)

    def t_to_r_sigmoid(self, t, iteration):
        d = self.n_train_iter // self.n_stages
        stage = iteration // d
        adj = 1 + self.k * torch.sigmoid(-self.b * t)
        decay = 1 / self.q ** (stage+1)
        ratio = 1 - decay * adj
        r = t * ratio
        return torch.clamp(r, min=0)

    def sample(self, num_samples, iteration):
        rnd_normal = torch.randn([num_samples]).cuda()
        t = (rnd_normal * self.std + self.mu).exp()
        r = self.t_to_r(t, iteration)
        return r, t