import argparse
import os
import torch
import torch.nn as nn
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.SelfExpressive import SelfExpressive
from utils.DataPrepare import prepare_data_YaleB, prepare_data_orl, prepare_data_coil20, prepare_data_coil100


class Generator(nn.Module):
    def __init__(self, args, batch_size, n_hidden, kernel_size):
        super(Generator, self).__init__()
        self.args = args
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.kernel_size = kernel_size
        self.encoder = Encoder(self.n_hidden, self.kernel_size)
        self.decoder = Decoder(self.n_hidden, self.kernel_size)
        self.self_expressive = SelfExpressive(self.batch_size)

    def forward_pretrain(self, x):
        latent = self.encoder.forward(x)
        x_r_pre = self.decoder.forward(latent)
        return x_r_pre

    def forward(self, x):
        latent = self.encoder.forward(x)
        z, z_c, Coef = self.self_expressive.forward(latent)
        latent_c = z_c.reshape(latent.shape)
        x_r = self.decoder.forward(latent_c)
        return x_r, Coef, z, z_c

    def save_model(self, model):
        torch.save(model, "./run_orl")

    def restore(self):
        model = torch.load("./run_orl")
        print("model restored")
        return model







