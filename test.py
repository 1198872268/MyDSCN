import argparse
import os
import torch
from utils.DataPrepare import *
from model.train import *
from tmodel.TGenerator import TGenerator

parser = argparse.ArgumentParser()

# parser.add_argument('name')
parser.add_argument('--name', type=str, default="run")
parser.add_argument('--lambda1', type=float, default=1.0)
# according to the paper, for yale lambda2=6.3096, for coil 20 and 100, lambda2=30.0, for orl, lambda2=0.2
parser.add_argument('--lambda2', type=float, default=1.0)  # sparsity cost on C
parser.add_argument('--lambda3', type=float, default=1.0)  # lambda on gan loss
parser.add_argument('--lambda4', type=float, default=0.00001)# lambda on AE L2 regularization
parser.add_argument('--m', type=float, default=0.1)  # lambda on AE L2 regularization


parser.add_argument('--lr',  type=float, default=2e-3)  # learning rate
# parser.add_argument('--lr2', type=float, default=2e-4)  # learning rate for discriminator and eqn3plus
parser.add_argument('--lr2', type=float, default=5e-5)  # learning rate for discriminator and eqn3plus

parser.add_argument('--pretrain', type=int, default=0)  # number of iterations of pretraining
parser.add_argument('--epochs', type=int, default=200)  # number of epochs to train on eqn3 and eqn3plus
# parser.add_argument('--enable-at', type=int, default=300)  # epoch at which to enable eqn3plus
parser.add_argument('--enable-at', type=int, default=100)  # epoch at which to enable eqn3plus
parser.add_argument('--dataset', type=str, default='orl', choices=['yaleb', 'orl', 'coil20', 'coil100'])
parser.add_argument('--interval', type=int, default=10)
parser.add_argument('--interval2', type=int, default=1)
parser.add_argument('--bound', type=float, default=0.02)  # discriminator weight clipping limit
parser.add_argument('--D-init', type=int, default=50)  # number of discriminators steps before eqn3plus starts
# parser.add_argument('--D-init', type=int, default=100)  # number of discriminators steps before eqn3plus starts
parser.add_argument('--D-steps', type=int, default=5)
parser.add_argument('--G-steps', type=int, default=1)

parser.add_argument('--save', action='store_true')  # save pretrained model
# parser.add_argument('--save', action='store_true')  # save pretrained model
"""
学习笔记  action——true 表示当 python main.py orl_run1   --save时,返回值为true,即可以执行保存操作
"""
parser.add_argument('--r', type=int, default=0)  # Nxr rxN, use 0 to default to NxN Coef
## new parameters
# parser.add_argument('--rank', type=int, default=10)  # dimension of the subspaces
parser.add_argument('--rank', type=int, default=30)  # dimension of the subspaces
parser.add_argument('--beta1', type=float, default=0.01)  # promote subspaces' difference
parser.add_argument('--beta2', type=float, default=0.010)  # promote org of subspaces' basis difference
parser.add_argument('--beta3', type=float, default=0.010)  # promote org of subspaces' basis difference

parser.add_argument('--stop-real', action='store_true')  # cut z_real path
parser.add_argument('--stationary', type=int, default=1)  # update z_real every so generator epochs

parser.add_argument('--submean',        action='store_true')
parser.add_argument('--proj-cluster',   action='store_true')
parser.add_argument('--usebn',          action='store_true')

parser.add_argument('--no-uni-norm',    action='store_true')
parser.add_argument('--one2one',        action='store_true')
#parser.add_argument('--alpha',          type=float, default=0.1)

parser.add_argument('--matfile',        default=None)
parser.add_argument('--imgmult',        type=float,     default=1.0)
parser.add_argument('--palpha',         type=float,     default=None)
parser.add_argument('--kernel-size',    type=int,       nargs='+',  default=None)

parser.add_argument('--s_tau1',        type=float,     default=1.01)
parser.add_argument('--s_lambda1',     type=float,     default=5.0)
parser.add_argument('--s_tau2',        type=float,     default=1.01)
parser.add_argument('--s_lambda2',     type=float,     default=4.0)
#parser.add_argument('--kernel-size',    type=int,       nargs='+',  default=None)
parser.add_argument('--degerate',       action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.name is not None and args.name != '', 'name of experiment must be specified'

    # prepare data
    folder = os.path.dirname(os.path.abspath(__file__))

    preparation_funcs = {
        'yaleb': prepare_data_YaleB,
        'orl': prepare_data_orl,
        'coil20': prepare_data_coil20,
        'coil100': prepare_data_coil100}

    assert args.dataset in preparation_funcs
    alpha, Img, Label, n_input, n_hidden, kernel_size, n_sample_perclass, disc_size, k, post_alpha, all_subjects, model_path = \
        preparation_funcs[args.dataset](folder)
    print(Img.shape)
    Img = torch.tensor(Img * args.imgmult, dtype=torch.float)
    post_alpha = args.palpha or post_alpha
    logs_path = os.path.join(folder, 'logs', args.name)
    restore_path = model_path

    Img = Img.cuda()

    batch_size = Img.shape[0]
    generator = TGenerator(args, batch_size, n_hidden=n_hidden, kernel_size=kernel_size).cuda()

    x_r_pre = generator.forward_pretrain(Img)


