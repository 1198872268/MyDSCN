import torch
import torch.nn as nn
from utils.Tprod import tprod,tcat


class TSelfexpressive(nn.Module):
    #  z → 64*1440*64  coef: 1440 * 1440 *64
    def __init__(self, batch_size):
        super(TSelfexpressive, self).__init__()
        self.batch_size = batch_size
        # Coef = 1.0e-4 * torch.ones((int(self.batch_size), int(self.batch_size)), dtype=torch.float32, requires_grad=True)
        Coef = 1.0e-4 * torch.ones((int(self.batch_size), int(self.batch_size), 64), dtype=torch.float32,
                                   requires_grad=True)
        self.Coef = nn.Parameter(Coef)

    def forward(self, z):
        Coef = self.Coef
        z_c = tprod(z, Coef)
        return z_c, Coef