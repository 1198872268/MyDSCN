import torch.nn as nn
from model.Pad import *


class Decoder(nn.Module):
    def __init__(self, n_hidden, kernel_size, reuse=False):
        super(Decoder, self).__init__()
        self.n_hidden = n_hidden
        self.kernel_size = kernel_size
        self.ConvT = self.decoder()

    def decoder(self):
        n_hidden = list(reversed([1] + self.n_hidden))
        net = nn.Sequential()
        for i, k_size in enumerate(reversed(self.kernel_size)):
            name = "convT_{}".format(i)
            if i != len(self.n_hidden) + 1:
                net.add_module(name, nn.Sequential(
                    nn.ConvTranspose2d(n_hidden[i], n_hidden[i+1], kernel_size=k_size, stride=[2, 2]),
                    ConvTranspose2dSamePad(k_size, 2),
                    nn.ReLU()
                ))
            else:
                net.add_module(name, nn.Sequential(
                    nn.ConvTranspose2d(n_hidden[i], 1, kernel_size=k_size, stride=[2, 2], padding=k_size // 2, output_padding=1,
                                       padding_mode='zeros')
                ))
        return net

    def forward(self, z):
        x_r = self.ConvT(z)
        # shapes = list(reversed(shapes))
        # n_hidden = list(reversed([1] + self.n_hidden))
        # net = self.ConvT
        # x_r = z
        # for i in net:
        #     x_r = i(x_r)
        return x_r





    # Building the decoder
    # def decoder1(self, z, shapes, reuse=False):
    #     # Encoder Hidden layer with sigmoid activation #1
    #     input = z
    #     n_hidden = list(reversed([1] + self.n_hidden))
    #     shapes = list(reversed(shapes))
    #     for i, k_size in enumerate(reversed(kernel_size)):
    #         with tf.variable_scope('', reuse=reuse):
    #             w = tf.get_variable('dec_w{}'.format(i), shape=[k_size, k_size, n_hidden[i + 1], n_hidden[i]],
    #                                     initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
    #             b = tf.get_variable('dec_b{}'.format(i), shape=[n_hidden[i + 1]],
    #                                     initializer=tf.zeros_initializer())
    #             dec_i = tf.nn.conv2d_transpose(input, w, tf.stack(
    #                     [tf.shape(self.x)[0], shapes[i][1], shapes[i][2], shapes[i][3]]),
    #                                                strides=[1, 2, 2, 1], padding='SAME')
    #             dec_i = tf.add(dec_i, b)
    #             if i != len(self.n_hidden) - 1:
    #                     dec_i = tf.nn.relu(dec_i)
    #             input = dec_i
    #     return input




