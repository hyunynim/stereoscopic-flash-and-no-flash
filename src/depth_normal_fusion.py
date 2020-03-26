import numpy as np
import scipy
from utils import get_sparse_matrix_operator
from scipy.sparse.linalg import lsqr


def depth_normal_fusion(z_map, n, mask, K, lamda, ratio=2):
    imask = np.flatnonzero(mask)
    z_flaten = z_map.ravel()[imask]
    npix = len(z_flaten)
    n1 = n[:npix]
    n2 = n[npix:npix * 2]
    n3 = n[npix * 2:]

    x0_in_world = K[0, 2] / ratio
    y0_in_world = K[1, 2] / ratio
    fx = K[0, 0]
    fy = K[1, 1]
    H, W = mask.shape

    yy, xx = np.meshgrid(np.arange(z_map.shape[1]), np.arange(z_map.shape[0]))
    xx, yy = xx.astype(np.float64), yy.astype(np.float64)
    xx = xx[::-1, :]
    xx -= (H - y0_in_world)
    yy -= (W - x0_in_world)
    xx_flatten = xx.ravel()[imask]
    yy_flatten = yy.ravel()[imask]


    D_h, D_v, ind2sub = get_sparse_matrix_operator(mask)
    D_x, D_y = - D_v, D_h

    Du1 = D_x.multiply(xx_flatten[:, None]) / fx + scipy.sparse.identity(npix, format="csr") / fx
    Du2 = D_x.multiply(yy_flatten[:, None]) / fy
    Du3 = D_x / ratio

    Dv1 = D_y.multiply(xx_flatten[:, None]) / fx
    Dv2 = D_y.multiply(yy_flatten[:, None]) / fy + scipy.sparse.identity(npix, format="csr") / fy
    Dv3 = D_y / ratio

    Du = Du1.multiply(n1[:, None]) + Du2.multiply(n2[:, None]) + Du3.multiply(n3[:, None])
    Dv = Dv1.multiply(n1[:, None]) + Dv2.multiply(n2[:, None]) + Dv3.multiply(n3[:, None])

    A = scipy.sparse.vstack([lamda * scipy.sparse.identity(npix, format="csr"),
                             (1 - lamda) * Du,
                             (1 - lamda) * Dv])
    b = np.concatenate([lamda * z_flaten,
                        np.zeros_like(z_flaten),
                        np.zeros_like(z_flaten)])
    z = lsqr(A, b, iter_lim=50, show=False)[0]
    return z
