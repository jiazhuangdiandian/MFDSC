import numpy as np
from sklearn import cluster
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

nmi = normalized_mutual_info_score
ami = adjusted_mutual_info_score
ari = adjusted_rand_score

# def purity(labels_true, labels_pred):
#     clusters = np.unique(labels_pred)
#     labels_true = np.reshape(labels_true, (-1, 1))
#     labels_pred = np.reshape(labels_pred, (-1, 1))
#     count = []
#     for c in clusters:
#         idx = np.where(labels_pred == c)[0]
#         labels_tmp = labels_true[idx, :].reshape(-1)
#         count.append(np.bincount(labels_tmp).max())
#     return np.sum(count) / labels_true.shape[0]

def purity(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    count = 0

    for cluster in clusters:
        mask = (labels_pred == cluster)
        labels_in_cluster = labels_true[mask]
        most_frequent_label = np.bincount(labels_in_cluster).argmax()
        count += np.sum(labels_in_cluster == most_frequent_label)

    return count / len(labels_true)

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


def err_rate(gt_s, s):
    return 1.0 - acc(gt_s, s)


def thrC(C, alpha):
    if alpha < 1:
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
                if csum > alpha * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d, ro):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    n = C.shape[0]
    C = 0.5 * (C + C.T)
    # C = C - np.diag(np.diag(C)) + np.eye(n, n)  # good for coil20, bad for orl/ORL/Umist
    r = d * K + 1
    # U, S, _ = svds(C, r, ncv=2*r + 1, v0=np.ones(n))
    U, S, _ = svds(C, r, v0=np.ones(n))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** ro)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L)
    return grp, L


def spectral_clustering(C, K, d, alpha, ro):
    # print('C.shape:',C.shape)
    C = thrC(C, alpha)
    y, _ = post_proC(C, K, d, ro)
    # print('y.shape:', y.shape)
    return y


def visualize(Img,Label,CAE=None,filep=None):
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax1 = fig.add_subplot(111)
    n = Img.shape[0]
    if CAE is not None:
        bs = CAE.batch_size
        Z = CAE.transform(Img[:bs,:])
        Z = np.zeros([Img.shape[0], Z.shape[1]])
        for i in range(Z.shape[0] // bs):
            Z[i * bs:(i + 1) * bs, :] = CAE.transform(Img[i * bs:(i + 1) * bs, :])
        if Z.shape[0] % bs > 0:
            Z[-bs:, :] = CAE.transform(Img[-bs:, :])
    else:
        Z = Img
    Z_emb = TSNE(n_components=2).fit_transform(Z, Label)
    lbs = np.unique(Label)

    for ii in range(len(lbs)):
        Z_embi = Z_emb[[i for i in range(n) if Label[i] == lbs[ii]]].transpose()
        ax1.scatter(Z_embi[0], Z_embi[1], color=colors[ii % 10], marker=marks[ii // 10], label=str(ii), s=20) #s=3
    ax1.legend()

    # plt.savefig('umist-1.svg', format='svg')
    plt.show()
colors = ['cyan', 'lime', 'r', 'royalblue', 'm', 'orange', 'gold','deepskyblue', 'greenyellow','fuchsia']
marks = [ 'o','o','o','o','o','o','o','o','o','o']