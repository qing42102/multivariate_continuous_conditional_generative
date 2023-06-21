import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Encoder(nn.Module):
    def __init__(self, shape, latent_dim=16, conditional_dim=0):
        super(Encoder, self).__init__()
        c, h, w = shape

        self.latent_dim = latent_dim
        self.conditional_dim = conditional_dim
        self.encode = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(start_dim=1),
        )

        self.calc_mean = nn.Linear(1024 + conditional_dim, self.latent_dim)
        self.calc_logvar = nn.Linear(1024 + conditional_dim, self.latent_dim)

    def forward(self, x, y=None):
        z = self.encode(x)

        if y is None:
            input = z
        else:
            input = torch.cat((z, y), dim=1)

        mean = self.calc_mean(input)
        logvar = self.calc_logvar(input)

        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, shape, latent_dim=16, conditional_dim=0):
        super(Decoder, self).__init__()
        c, w, h = shape
        self.shape = shape
        self.decode = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(64, 4, 4)),
            nn.Upsample(size=8),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Upsample(size=24),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1),
            nn.Sigmoid(),
        )

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim + conditional_dim, 1024), nn.ReLU(True)
        )

    def forward(self, z, y=None):
        c, w, h = self.shape
        if y is None:
            input = z
        else:
            input = torch.cat((z, y), dim=1)

        input = self.decoder_lin(input)
        image = self.decode(input)
        image = image.view(-1, c, w, h)

        return image


class conditional_VAE(nn.Module):
    def __init__(self, shape, latent_dim=16, conditional_dim=0):
        super(conditional_VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(shape, latent_dim, conditional_dim=conditional_dim)
        self.decoder = Decoder(shape, latent_dim, conditional_dim=conditional_dim)

    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x, y=None):
        mean, logvar = self.encoder(x, y)
        z = self.sampling(mean, logvar)
        return self.decoder(z, y), mean, logvar

    def generate(self, y):
        num_data = y.shape[0]
        z = torch.randn((num_data, self.latent_dim)).to(device)
        image = self.decoder(z, y)

        return image


def loss_function(X, X_hat, mean, logvar):
    MSE_loss = nn.MSELoss(reduction="sum")
    reconstruction_loss = MSE_loss(X_hat, X)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
    return reconstruction_loss + KL_divergence
