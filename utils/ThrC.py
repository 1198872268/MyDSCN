import numpy as np
import torch

# 保留数值较大的系数
def thrC(C, ro):
    C = C.cpu().detach().numpy()
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C
    return Cp


def affinity_matrix(coef):
    n3 = coef.shape[2]
    coef_t = torch.permute(coef, [1, 0, 2])
    c = abs(coef_t) + abs(coef)
    c = torch.sum(c, dim=2)
    return c

if __name__ == '__main__':
    b = [i for i in range(8)]
    c = torch.tensor(b)
    c = torch.reshape(c, [2, 2, 2])
    print(affinity_matrix(c))

