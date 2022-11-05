import os
import scipy.io as sio
import numpy as np

def prepare_data_YaleB(folder):
    # load face images and labels
    mat = sio.loadmat(os.path.join(folder, 'Yale.mat'))
    img = mat['Y']

    # Reorganize data a bit, put images into Img, and labels into Label
    I = []
    Label = []
    for i in range(img.shape[2]):       # i-th subject
        for j in range(img.shape[1]):   # j-th picture of i-th subject
            temp = np.reshape(img[:,j,i],[42,48])
            Label.append(i)
            I.append(temp)
    I = np.array(I)
    Label = np.array(Label[:])
    Img = np.transpose(I,[0,2,1])
    Img = np.expand_dims(Img[:],3)

    # constants
    n_input = [48,42]
    n_hidden = [10,20,30]
    kernel_size = [5,3,3]
    n_sample_perclass = 64
    disc_size = [200,50,1]
    # tunable numbers
    k=10
    post_alpha=3.5
    alpha = 0.1

    all_subjects = [38] # number of subjects to use in experiment
    model_path = os.path.join(folder, 'yale-model.ckpt')
    return alpha, Img, Label, n_input, n_hidden, kernel_size, n_sample_perclass, disc_size, k, post_alpha, all_subjects, model_path


def prepare_data_orl(folder):
    # 标准化情况下：loss_eqn3 = 10000000*loss_recon + 100 * loss_sparsity + 5 * loss_selfexpress
    mat = sio.loadmat('C://Users//xx//Desktop//常用文件//研//入门论文//论文代码//MyDASC//data/ORL(2).mat')
    Label = mat['gnd'].reshape(-1).astype(np.int32)
    Img = mat['fea'].reshape(-1, 1, 32, 32)
    # Img = normalize_data(Img)
    # Img = Img/256
    # constants
    n_input = [32, 32]
    n_hidden = [3, 3, 5]
    kernel_size = [3, 3, 3]
    n_sample_perclass = Img.shape[0] / 40
    disc_size = [50, 1]
    # tunable numbers
    k = 12
    dim_subspace = 3
    post_alpha = 1.0  # Laplacian parameter
    alpha = 0.2

    all_subjects = 40
    model_path = os.path.join(folder, 'orl-model15.ckpt')
    return alpha, Img, Label, n_input, n_hidden, kernel_size, n_sample_perclass, disc_size, k, dim_subspace,  post_alpha, all_subjects, model_path


def prepare_data_coil20(folder):
    mat = sio.loadmat('C://Users//xx//Desktop//常用文件//研//入门论文//论文代码//MyDASC//data/COIL20.mat')
    Label = mat['gnd'].reshape(-1).astype(np.int32)
    Img = mat['fea'].reshape(-1, 1, 32, 32)
    #Img = normalize_data(Img)

    # constants
    n_input = [32, 32]
    n_hidden = [15]
    kernel_size = [3]
    n_sample_perclass = Img.shape[0] / 20
    disc_size = [50, 1]
    # tunable numbers
    k = 12            # svds parameter
    post_alpha = 8.0  # Laplacian parameter
    alpha = 0.04

    all_subjects = 20
    model_path = os.path.join(folder, 'coil20-model15.ckpt')
    return alpha, Img, Label, n_input, n_hidden, kernel_size, n_sample_perclass, disc_size, k, post_alpha, all_subjects, model_path


def prepare_data_coil100(folder):
    mat = sio.loadmat(os.path.join(folder, 'COLT100.mat'))
    Label = mat['gnd'].reshape(-1).astype(np.int32) # 1440
    Img = mat['fea'].reshape(-1, 32, 32, 1)

    # constants
    n_input  = [32, 32]
    n_hidden = [50]
    kernel_size = [5]
    n_sample_perclass = Img.shape[0] / 100
    disc_size = [50,1]
    # tunable numbers
    k=12            # svds parameter
    post_alpha=8.0  # Laplacian parameter
    alpha = 0.04

    all_subjects=[100]
    model_path  = os.path.join(folder, 'coil100-model50.ckpt')
    return alpha, Img, Label, n_input, n_hidden, kernel_size, n_sample_perclass, disc_size, k, post_alpha, all_subjects, model_path