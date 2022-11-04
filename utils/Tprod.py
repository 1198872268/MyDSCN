import torch


def tprod(a, b):
    n1, _, n3 = a.shape
    l = b.shape[1]
    a = torch.fft.fft(a, dim=2)
    b = torch.fft.fft(b, dim=2)
    c = torch.zeros(n1, l, n3)
    for i in range(n3):
        c[:, :, i] = torch.matmul(a[:, :, i], b[:, :, i])
    c = torch.fft.ifft(c, dim=2).real   # dct
    return c


# 将encoder所得的多通道合为一个矩阵，并将每个x作为一个侧切片
def tcat(latent):
    n, channel, w, h = latent.shape
    r = int(pow(channel, 0.5))
    z = torch.empty(n, r*w, r*w)
    for i in range(n):
        for j in range(channel):
            z[i, j // (int(pow(channel, 0.5))) * w:((j // int(pow(channel, 0.5)) + 1) * w)
            , j % (int(pow(channel, 0.5))) * w:(j % (int(pow(channel, 0.5))) + 1) * w] = latent[i, j, :, :]
    z = torch.permute(z, [1, 0, 2])
    return z


# 将自表示层输出结果恢复为 可作为decoder输入的shape
def tsplit(z_c, channel):
    channel = channel[-1]
    z_c = torch.permute(z_c, [1, 0, 2])
    n, r = z_c.shape[0], z_c.shape[1]
    size = int(r / pow(channel, 0.5))
    zr = torch.empty(n, channel, size, size)
    for i in range(n):
        for j in range(channel):
            x = z_c[i, j // (int(pow(channel, 0.5))) * size:((j // int(pow(channel, 0.5)) + 1) * size)
            , j % (int(pow(channel, 0.5))) * size:(j % (int(pow(channel, 0.5))) + 1) * size]
            zr[i, j, :, :] = x
    return zr


if __name__ == '__main__':
    # b = [i for i in range(108)]
    # c = torch.tensor(b)
    # c = torch.reshape(c, [3, 4, 3, 3])
    # y = tcat(c)
    # z = torch.ones(3, 3, 6)
    A = torch.zeros([3, 2, 2])
    B = torch.zeros([2, 1, 2])

    A[:, :, 0] = torch.tensor([[1, 0], [0, 2], [-1, 3]])
    A[:, :, 1] = torch.tensor([[-2, 1], [-2, 7], [0, -1]])

    B[:, :, 0] = torch.tensor([[3], [-1]])
    B[:, :, 1] = torch.tensor([[-2], [-3]])
    x = tprod(A, B)
    print(A)
    print(B)
    print(x)

