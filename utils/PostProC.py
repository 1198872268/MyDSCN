import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1  # K=38, d=10
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    # U, S, _ = svd_cuda(C, allocator=mem_pool)
    # take U and S from GPU
    # U = U[:, :r].get()
    # S = S[:r].get()
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    #Z = Z * (Z > 0)
    L = np.abs(np.abs(Z) ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L)  # +1
    return grp, L