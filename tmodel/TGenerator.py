import argparse
import os
import torch
import torch.nn as nn
from model.Encoder import Encoder
from model.Decoder import Decoder
from tmodel.TSelfexpress import TSelfexpressive
from utils.Tprod import tcat,tsplit


class TGenerator(nn.Module):
    def __init__(self, args, batch_size, n_hidden, kernel_size):
        super(TGenerator, self).__init__()
        self.args = args
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.kernel_size = kernel_size
        self.encoder = Encoder(self.n_hidden, self.kernel_size)
        self.decoder = Decoder(self.n_hidden, self.kernel_size)
        self.self_expressive = TSelfexpressive(self.batch_size)

    def forward_pretrain(self, x):
        latent, shapes = self.encoder.forward(x)
        x_r_pre = self.decoder.forward(latent, shapes)
        return x_r_pre

    def forward(self, x, n_hidden):
        latent, shapes = self.encoder.forward(x)
        z = tcat(latent)
        z = z.cuda()
        z_c, Coef = self.self_expressive.forward(z)
        z_c = z_c.cuda()
        latent_c = tsplit(z_c, n_hidden).cuda()
        x_r = self.decoder.forward(latent_c, shapes)
        return x_r, Coef, z, z_c
# z: latent变形为z后输入selfexpress层，  z_c：z*coef， latent_c：z_c变为decoder输入的形状

    def save_model(self, model):
        torch.save(model, "./trun_orl")

    def restore(self):
        model = torch.load("./trun_orl")
        print("model restored")
        return model







