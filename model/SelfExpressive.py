import torch
import torch.nn as nn


class SelfExpressive(nn.Module):
    def __init__(self, batch_size):
        super(SelfExpressive, self).__init__()
        self.batch_size = batch_size
        Coef = 1.0e-8 * torch.ones((int(self.batch_size), int(self.batch_size)), dtype=torch.float32, requires_grad=True)
        self.Coef = nn.Parameter(Coef)

    def forward(self, latent):
        z = latent.reshape([int(self.batch_size), -1])
        Coef = self.Coef
        z_c = torch.matmul(Coef, z)
        return z, z_c, Coef