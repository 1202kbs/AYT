import numpy as np

def get_step_schedule(cfg):
    if cfg['name'] == 'constant':
        return Constant(s0=cfg['s0'],s1=cfg['s1'])
    elif cfg['name'] == 'exponential':
        return Exponential(s0=cfg['s0'],s1=cfg['s1'])

class Constant:
    def __init__(self,s0,s1):
        self.s0 = s0
        self.s1 = s1

    def __call__(self, k, K):
        return self.s0

class Exponential:

    def __init__(self,s0,s1):
        self.s0 = s0
        self.s1 = s1

    def __call__(self, k, K):
        Kp = int(K / (np.log2(int(self.s1/self.s0))+1))
        return min(self.s0*2**(int(k/Kp)),self.s1)