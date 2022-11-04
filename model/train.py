import torch
import numpy as np
from utils.ThrC import thrC,affinity_matrix
from utils.PostProC import post_proC
from utils.ErrRate import err_rate
import time
import torch.nn.functional as F


def pretrain(generator, x, learning_rate, weight_decay, num_epochs):
    optimizer = torch.optim.Adam(generator.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(1, num_epochs+1):
        minibatch_size = 128
        indices = np.random.permutation(x.shape[0])[:minibatch_size]
        minibatch = x[indices]  # pretrain with random mini-batch
        x_r_pre = generator.forward_pretrain(minibatch)
        loss_recon_pre = 0.5 * torch.sum(torch.pow(torch.subtract(x_r_pre, minibatch), 2.0))
        optimizer.zero_grad()
        loss_recon_pre.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('pretraining epoch {}, cost: {}'.format(epoch, loss_recon_pre / float(minibatch_size)))
    return loss_recon_pre


def train_equ3(generator, x, label, learning_rate, weight_decay, enable_at, lambda1, lambda2, batch_size, n_hidden, k=12,
                dim_subspace=8, n_class=20, post_alpha=8, ro=0.04):
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(generator.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    max_acc, best_epoch = 0, 0
    for epoch in range(1, enable_at+1):
        x_r, coef, z, z_c = generator.forward(x)
        # loss_recon = 0.5 * torch.sum(torch.pow(torch.subtract(x_r, x), 2.0))
        # loss_sparsity = torch.sum(torch.abs(coef))
        # loss_selfexpress = 0.5 * torch.sum(torch.pow(torch.subtract(z_c, z), 2.0))
        loss_recon = F.mse_loss(x_r, x, reduction='sum')
        loss_selfexpress = F.mse_loss(z_c, z, reduction='sum')
        loss_sparsity = torch.sum(torch.pow(coef, 2))#
        loss_eqn3 = 10*loss_recon + lambda1 * loss_sparsity + lambda2 * loss_selfexpress  # + self.loss_aereg
        optimizer.zero_grad()
        loss_eqn3.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print("epoch: %.1d" % epoch, "cost: %.8f" % (loss_eqn3 / float(batch_size)), "loss_sparsity:%.8f" % loss_sparsity, "loss_selfexpress:%.8f" % loss_selfexpress, "loss_recon:%.8f" % loss_recon)
            coef = thrC(coef, ro)
            t_begin = time.time()
            y_x_new, _ = post_proC(coef, n_class, dim_subspace, post_alpha)
            if (len(set(list(np.squeeze(y_x_new)))) == n_class) or (epoch == 10):   #已经修改
                y_x = y_x_new
            else:
                print('================================================')
                print('Warning: clustering produced empty clusters')
                print('================================================')
                print("class_num:{}".format(len(set(list(np.squeeze(y_x_new))))))
            missrate_x = err_rate(label, y_x)
            t_end = time.time()
            acc_x = 1 - missrate_x
            if max_acc < acc_x:
                max_acc = acc_x
                best_epoch = epoch
            print("accuracy: {}".format(acc_x))
            print('post processing time: {}'.format(t_end - t_begin))
    print("max_acc:{},best_epoch:{}".format(max_acc, best_epoch))
    return loss_eqn3, coef


def t_train_equ3(generator, x, label, learning_rate, weight_decay, enable_at, lambda1, lambda2, batch_size, n_hidden, k=3, n_class=20, post_alpha=8, ro=0.04):
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(generator.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(1, enable_at+1):
        x_r, coef, z, z_c = generator.forward(x, n_hidden)
        # coef = affinity_matrix(coef)
        coef = coef[:, :, 0]
        loss_recon = 0.5 * torch.sum(torch.pow(torch.subtract(x_r, x), 2.0))
        loss_sparsity = torch.sum(torch.abs(coef))
        loss_selfexpress = 0.5 * torch.sum(torch.pow(torch.subtract(z_c, z), 2.0))
        loss_eqn3 = 3*loss_recon + lambda1 * loss_sparsity + lambda2 * loss_selfexpress  # + self.loss_aereg
        # loss_eqn3 = loss_eqn3/batch_size
        optimizer.zero_grad()
        loss_eqn3.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print("epoch: %.1d" % epoch, "cost: %.8f" % (loss_eqn3/batch_size), "loss_sparsity:%.8f" % loss_sparsity, "loss_selfexpress:%.8f" % loss_selfexpress, "loss_recon:%.8f" % loss_recon)
            coef = thrC(coef[:, :, 0], ro)
            t_begin = time.time()
            y_x_new, _ = post_proC(coef, n_class, k, post_alpha)
            if (len(set(list(np.squeeze(y_x_new)))) == n_class) or (epoch == 10):   #已经修改
                y_x = y_x_new
            else:
                print('================================================')
                print('Warning: clustering produced empty clusters')
                print('================================================')
                print("class_num:{}".format(len(set(list(np.squeeze(y_x_new))))))
            missrate_x = err_rate(label, y_x)
            t_end = time.time()
            acc_x = 1 - missrate_x
            print("accuracy: {}".format(acc_x))
            print('post processing time: {}'.format(t_end - t_begin))
    return loss_eqn3, coef

