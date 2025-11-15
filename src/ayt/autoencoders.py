import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, dropout=0.0):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(2, stride=2),
            # nn.Conv2d(64, 128, 3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Dropout2d(p=dropout),
            # nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            # nn.Dropout2d(p=dropout),
            nn.ConvTranspose2d(64, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, 3, stride=1, padding=1, output_padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def get_ae(cfg):
    return Autoencoder(dropout=cfg['dropout'])