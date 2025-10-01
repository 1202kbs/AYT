from torch.utils.data import DataLoader
from ayt.constants import DATA_ROOT
from torchvision import datasets

import torchvision.transforms as transforms

def get_dataset(cfg):
    
    img_tfs = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5]*cfg['channels'], std=[0.5]*cfg['channels'])])

    if cfg['name'] == 'mnist':
        return datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=img_tfs)
    elif cfg['name'] == 'fmnist':
        return datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True, transform=img_tfs)
    elif cfg['name'] == 'kmnist':
        return datasets.KMNIST(root=DATA_ROOT, train=True, download=True, transform=img_tfs)
    elif cfg['name'] == 'cifar10':
        return datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=img_tfs)
    elif cfg['name'] == 'cifar10_test':
        return datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=img_tfs)