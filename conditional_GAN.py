import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, label_dim):
        super(Generator, self).__init__()

        self.label_embedding = nn.Sequential(
            nn.Linear(label_dim, 32),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(32, 1, 1)),
        )

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + 32, 32 * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(32 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(32 * 4, 32 * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32 * 2, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, noise_vector, label):
        label_output = self.label_embedding(label)

        concat = torch.cat((noise_vector, label_output), dim=1)
        image = self.model(concat)
        return image


class Discriminator(nn.Module):
    def __init__(self, label_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(4, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 2, 32 * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

        self.label_embedding = nn.Sequential(
            nn.Linear(label_dim, 784),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(1, 28, 28)),
        )

    def forward(self, image, label):
        label_output = self.label_embedding(label)
        concat = torch.cat((image, label_output), dim=1)
        output = self.model(concat)
        output = torch.squeeze(output)
        return output


def generator_loss(label, fake_output):
    binary_cross_entropy = nn.BCELoss()
    gen_loss = binary_cross_entropy(label, fake_output)
    return gen_loss


def discriminator_loss(label, output):
    binary_cross_entropy = nn.BCELoss()
    disc_loss = binary_cross_entropy(label, output)
    return disc_loss
