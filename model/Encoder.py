import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, n_hidden, kernel_size):
        super(Encoder, self).__init__()
        self.n_hidden = n_hidden
        self.kernel_size = kernel_size
        self.conv = self.encoder()

    def encoder(self):
        n_hidden = [1] + self.n_hidden
        net = nn.Sequential()
        for i, k_size in enumerate(self.kernel_size):
            name = "conv_{}".format(i)
            net.add_module(name, nn.Sequential(
                            nn.Conv2d(n_hidden[i], n_hidden[i+1], kernel_size=k_size, stride=(2, 2), padding=k_size // 2, padding_mode='zeros'),
                            nn.ReLU()
                            ))
        return net

    def forward(self, x):
        net = self.conv
        z = x
        shapes = []
        for i in net:
            shapes.append(list(z.shape))
            z = i(z)
        return z, shapes










    # def encoder1(self, x):
    #     shapes = []
    #     n_hidden = [1] + self.n_hidden
    #     input = x
    #     for i, k_size in enumerate(self.kernel_size):
    #         w = tf.get_variable('enc_w{}'.format(i), shape=[k_size, k_size, n_hidden[i], n_hidden[i + 1]],
    #                             initializer=layers.xavier_initializer_conv2d(), regularizer=self.reg)
    #         # 卷积核
    #         b = tf.get_variable('enc_b{}'.format(i), shape=[n_hidden[i + 1]], initializer=tf.zeros_initializer())
    #         shapes.append(input.get_shape().as_list())
    #         enc_i = tf.nn.conv2d(input, w, strides=[1, 2, 2, 1], padding='SAME')
    #         enc_i = tf.nn.bias_add(enc_i, b)
    #         enc_i = tf.nn.relu(enc_i)
    #         input = enc_i
    #     return input, shapes




