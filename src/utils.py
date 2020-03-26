import os
import pickle
from glob import glob
from math import ceil, floor

import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image, ImageChops
from scipy.sparse import coo_matrix
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
from tqdm import tqdm

from lapalacian_eq_solver import *
from remove_outlier import outliner_removal

osp = os.path.join


def hide_all_plot(img, colorbar=True, fname=None, title="", vmin=0, vmax=10):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([1], [1])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(img, vmax=vmax, vmin=vmin)
    if colorbar:
        plt.colorbar()
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    ax.axis('off')
    ax.tick_params(axis=u'both', which=u'both', length=0)
    plt.title(title)
    if fname:
        plt.savefig(fname)
    else:
        plt.show()
    plt.clf()


def save_normal(fname, n_obj):
    cv2.imwrite(fname, swap_RB_channel((255 * (n_obj + 1) / 2).astype(np.uint8)))


def render_SH(l, rho=None, d=500, n=None, obj_mask=None):
    """
    if rho is not provided, assume it is 1.
    if n is not provided, assume its a sphere with radius d / 2.
    obj_mask should be provided along with n.
    n should be in world coordinate.
    :param l: lighting coefficient of shape (ch, 9)
    :return:
    """
    if l.ndim == 2:
        n_ch = l.shape[0]
    else:
        n_ch = 1
    if n is None:
        n, obj_mask, _ = generate_normal_map_and_depth(d)  # object coordinate
        n[..., 2] = -n[..., 2]
        temp0 = n[..., 0].copy()
        temp1 = n[..., 1].copy()
        n[..., 0], n[..., 1] = temp1, temp0  # world coordinate
    if rho is None:
        rho = np.ones_like(n)
    # Make image
    I = np.zeros((n.shape[0], n.shape[1], n_ch))
    NxNy = n[..., 0] * n[..., 1]
    NxNz = n[..., 0] * n[..., 2]
    NyNz = n[..., 1] * n[..., 2]
    for ch in range(n_ch):
        I[..., ch] = rho[..., ch] * (l[ch, 0] * n[..., 0] + l[ch, 1] * n[..., 1] + l[ch, 2] * n[..., 2] + l[ch, 3] + \
                                     l[ch, 4] * NxNy + l[ch, 5] * NxNz + l[ch, 6] * NyNz + \
                                     l[ch, 7] * (n[..., 0] ** 2 - n[..., 1] ** 2) + l[ch, 8] * (
                                                 3 * (n[..., 2] ** 2) - 1))
    I[~obj_mask] = np.NaN  # set background as 0
    return np.squeeze(I)


def normal_map_to_sh_map(n):
    '''
    transfer a normal map hw3 to the sperical harmonics map hw9
    :param n:
    :return:
    '''
    H, W, _ = n.shape
    nsh = np.zeros((H, W, 9))
    n1 = n[..., 0]
    n2 = n[..., 1]
    n3 = n[..., 2]
    nsh[..., 0] = n1
    nsh[..., 1] = n2
    nsh[..., 2] = n3
    nsh[..., 3] = np.ones_like(n1)
    nsh[..., 4] = n1 * n2
    nsh[..., 5] = n1 * n3
    nsh[..., 6] = n2 * n3
    nsh[..., 7] = n1 ** 2 - n2 ** 2
    nsh[..., 8] = 3 * n3 ** 2 - 1
    return nsh


def mkdir(data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


def swap_RB_channel(img):
    img = img.copy()
    temp = img[..., 0].copy()
    img[..., 0] = img[..., 2]
    img[..., 2] = temp
    return img


def data_loader(data_dir, ratio=2):
    obj_name = data_dir.split("/")[-1]

    mask = cv2.imread(osp(data_dir, "mask.png"), -1)
    if mask.ndim == 3:
        mask = mask[..., 0]
    img_size = mask.shape

    disparity_path = glob(osp(data_dir, "*disparity*"))[0]
    disparity = np.load(disparity_path)
    disp_size = disparity.shape

    camera_parameter_path = glob(osp(data_dir, "*camera*"))[0]
    with open(camera_parameter_path, "rb") as f:
        data = pickle.load(f)
    K_l, d_l, K_r, d_r, R, T, E, F = data["Intrinsic_M_l"], data["dist_l"], data["Intrinsic_M_r"], data["dist_r"], \
                                     data["R"], data["T"], data["E"], data["F"]
    R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(K_l, d_l, K_r, d_r, (img_size[1], img_size[0]), R, T,
                                                                flags=cv2.CALIB_ZERO_DISPARITY)
    Q[0, -1] /= ratio
    Q[1, -1] /= ratio
    xmap1, ymap1 = cv2.initUndistortRectifyMap(K_l, d_l, R1, P1, (img_size[1], img_size[0]), cv2.CV_32FC1)
    xmap2, ymap2 = cv2.initUndistortRectifyMap(K_r, d_r, R2, P2, (img_size[1], img_size[0]), cv2.CV_32FC1)

    mask_rectified = cv2.remap(mask, xmap1, ymap1, cv2.INTER_LANCZOS4)
    mask = cv2.resize(mask_rectified, (disp_size[1], disp_size[0]))
    mask = mask == 0

    flash_img_rgb_path = glob(osp(data_dir, "flash_rgb.npz"))[0]
    flash_img_rgb = np.load(flash_img_rgb_path)
    no_flash_img_rgb_path = glob(osp(data_dir, "no_flash_rgb.npz"))[0]
    no_flash_img_rgb = np.load(no_flash_img_rgb_path)

    flash_img_rgb = cv2.remap(flash_img_rgb, xmap1, ymap1, cv2.INTER_LANCZOS4)
    flash_img_rgb = cv2.resize(flash_img_rgb, (disp_size[1], disp_size[0]))

    no_flash_img_rgb = cv2.remap(no_flash_img_rgb, xmap1, ymap1, cv2.INTER_LANCZOS4)
    no_flash_img_rgb = cv2.resize(no_flash_img_rgb, (disp_size[1], disp_size[0]))

    disparity[~mask] = np.NaN
    disp_removed = outliner_removal(crop_image_by_mask(disparity, mask))
    bbox = crop_mask(mask)
    disparity[bbox[1]:bbox[3], bbox[0]:bbox[2]] = disp_removed

    nanmask = np.logical_and(mask, np.isnan(disparity))
    cropped_mask = crop_image_by_mask(nanmask, mask)
    cropped_disp = crop_image_by_mask(disparity, mask)

    dis_filled = laplacian_solver_for_Dirichlet_boundary_condition(cropped_disp.copy(), cropped_mask)
    temp = disparity.copy()

    disparity[bbox[1]:bbox[3], bbox[0]:bbox[2]] = dis_filled
    hole_mask = ~(np.isnan(temp) & ~np.isnan(disparity))

    z_points = cv2.reprojectImageTo3D(disparity, Q)
    z_points[..., 2] /= ratio
    z_points[~mask] = np.NaN
    z = z_points[..., 2].copy()

    nanmask = np.logical_and(mask, ~np.isnan(z))
    z_points[..., 0] -= np.nanmean(z_points[..., 0][nanmask])
    z_points[..., 1] -= np.nanmean(z_points[..., 1][nanmask])
    z_points[..., 2] -= np.nanmean(z_points[..., 2][nanmask])
    points_flatten = - z_points.reshape((-1, 3))[nanmask.flatten()]
    if not os.path.exists(osp(data_dir, "normal_PCA.npy")):
        tree = KDTree(points_flatten, leaf_size=1)
        # num_points x num_neighbours x num_features
        normal_list = []
        for i in tqdm(range(points_flatten.shape[0])):
            ind = tree.query_radius(points_flatten[i:i + 1], r=5)
            if ind[0].shape[0] < 3:
                n = np.array([0, 0, 1])
                normal_list.append(n)
                continue
            shifted_normal = points_flatten[ind[0]] - np.mean(points_flatten[ind[0]], axis=0, keepdims=1)
            u, s, vh = np.linalg.svd(shifted_normal, full_matrices=0)
            n = vh[-1, ...]
            if n[-1] < 0:
                n = -n
            normal_list.append(n)
        normal_map_flatten = np.asarray(normal_list)
        normal_map_flatten = normalize(normal_map_flatten, axis=1)
        normal_map_show = (normal_map_flatten + 1) / 2
        normal_map = np.ones((disp_size[0] * disp_size[1], 3), dtype=np.float32) / 2.  # make it filled with 0.5
        normal_map[nanmask.flatten()] = normal_map_show
        normal_map_obj = normal_map.reshape((disp_size[0], disp_size[1], 3))
        normal_map_obj[..., 0] = 1 - normal_map_obj[..., 0]
        np.save(osp(data_dir, "normal_PCA"), normal_map_obj)
    else:
        normal_map_obj = np.load(osp(data_dir, "normal_PCA.npy"))

    initial_normal_world = normal_map_obj * 2 - 1
    initial_normal_world = world_to_object(initial_normal_world)

    mf = cv2.cvtColor(swap_RB_channel(flash_img_rgb.astype(np.uint8)), cv2.COLOR_BGR2GRAY)
    mnf = cv2.cvtColor(swap_RB_channel(no_flash_img_rgb.astype(np.uint8)), cv2.COLOR_BGR2GRAY)
    m_r = mnf / (mf - mnf)  # 1500 X 2048

    m_r[m_r > 100] = np.NaN
    m_r[m_r < 0] = np.NaN
    m_r[~mask] = np.NaN
    mask = np.logical_and(mask, ~np.isnan(m_r))
    mask = np.logical_and(mask, z > 0)
    N_nan_mask = np.sum(np.isnan(initial_normal_world[:, :]), axis=-1).astype(np.bool)
    mask = np.logical_and(~N_nan_mask, mask)
    N_zero_mask = initial_normal_world[..., 2] == 0
    mask = np.logical_and(mask, ~N_zero_mask)
    mask = boundary_discarded_mask(mask)

    # no_flash_img_rgb[~mask] = np.NaN

    data = dict()
    data["z_init"] = z
    data["n_init"] = initial_normal_world
    data["n_init_obj"] = normal_map_obj  # for visualization
    data["mask"] = mask
    data["K"] = K_l
    data["mnf_rgb"] = no_flash_img_rgb
    data["mf_rgb"] = flash_img_rgb
    data["hole_mask"] = hole_mask
    data["pts"] = z_points
    data["pts_init"] = z_points
    return data


def clip_99_1_quantile(m, q=1):
    temp = m.copy()
    q1 = np.nanpercentile(m, q)
    q99 = np.nanpercentile(m, 100 - q)
    temp[temp < q1] = np.NaN
    temp[temp > q99] = np.NaN
    return temp


def clip_99_quantile(m, q=1):
    temp = m.copy()
    q99 = np.nanpercentile(m, 100 - q)
    temp[temp > q99] = np.NaN
    return temp


def generate_normal_map_and_depth(n):
    """
    Right-hand world coordinate. p = [x, y, sqrt(r^2 - (x - c_x)^2 - (y - c_y) ^ 2)].
    n = [z_x, z_y, 1] = [(x-c_x), (y - c_y), z] = [(x - c), (y - c), sqrt(r^2 - (x - c_x)^2 - (y - c_y) ^ 2)]
    :param n:
    :return:
    """
    # This is in world coordinate
    r = (n - 1) / 2
    II, JJ = np.meshgrid(range(n), range(n))  # x point to right, y point to bottom
    center_x = r
    center_y = r
    mask = r ** 2 - (II - center_x) ** 2 - (JJ - center_y) ** 2 < 1e-12
    z = np.sqrt(r ** 2 - (II - center_x) ** 2 - (JJ - center_y) ** 2)
    z[mask] = 0
    obj_mask = ~mask

    n = np.zeros((z.shape[0], z.shape[1], 3))
    n[..., 0] = (II - center_x)
    n[..., 1] = - (JJ - center_y)  # reverse y axis
    n[..., 2] = z
    n[mask] = [0, 0, 0]
    n = normalize_normal_map(n)

    z[mask] = np.NaN
    return n, obj_mask, z


def quantization(X, L, method="uniform"):
    nan_mask = np.isnan(X)
    X[nan_mask] = np.mean(X[~nan_mask])

    high = ceil(np.max(X))
    low = floor(np.min(X))
    qstep = (high - low) / L  # step size
    Q = np.floor((X - low) / qstep)
    low = low + qstep / 2
    Y = low + qstep * Q

    Y[nan_mask] = np.NaN
    return Y


def world_to_object(n):
    no = n.copy()
    no[..., 2] = -no[..., 2]
    temp0 = no[..., 0].copy()
    temp1 = no[..., 1].copy()
    no[..., 1] = temp0
    no[..., 0] = temp1
    return no


def mae_map(n, gt_n):
    return np.rad2deg(np.arccos(np.sum(np.multiply(n, gt_n), axis=-1)))


def normalize_normal_map(N):
    """
    N is a unnormalized normal map of shape H_W_3. Normalize N across the third dimension.
    :param N:
    :return:
    """
    H, W, C = N.shape
    N = np.reshape(N, (-1, C))
    N = normalize(N, axis=1)
    N = np.reshape(N, (H, W, C))
    return N


def detect_boundary_points(mask):
    """
    a boolean mask. True means there is an object.
    :param mask:
    :return:
    """
    indices = np.where(mask)
    ind_set = {i for i in zip(indices[0], indices[1])}
    boundary_set = set()
    for i in ind_set:
        left = (i[0], i[1] - 1)
        right = (i[0], i[1] + 1)
        top = (i[0] - 1, i[1])
        bottom = (i[0] + 1, i[1])
        if left not in ind_set or right not in ind_set or top not in ind_set or bottom not in ind_set:
            boundary_set.add(i)
    return list(boundary_set)


def recover_normal_map(zx, zy, mask):
    H, W = mask.shape
    n = - np.ones((H*W, 3))
    imask = np.flatnonzero(mask)
    n[imask, 0] = zx
    n[imask, 1] = zy
    n = np.reshape(n, (H, W, 3))
    n = normalize_normal_map(n)
    return n


def boundary_discarded_mask(mask):
    """
    a boolean mask. True means there is an object.
    :param mask:
    :return:
    """
    indices = np.where(mask)
    ind_set = {i for i in zip(indices[0], indices[1])}
    bd_mask = mask.copy()
    for i in ind_set:
        left = (i[0], i[1] - 1)
        right = (i[0], i[1] + 1)
        top = (i[0] - 1, i[1])
        bottom = (i[0] + 1, i[1])
        if left not in ind_set or right not in ind_set or top not in ind_set or bottom not in ind_set:
            bd_mask[i] = False
    return bd_mask


def normal_to_SH(n):
    """

    :param n:
    :return:
    """
    H, W, _ = n.shape
    N_SH_gt = np.zeros((H, W, 9))
    N_SH_gt[..., 0] = n[..., 0]
    N_SH_gt[..., 1] = n[..., 1]
    N_SH_gt[..., 2] = n[..., 2]
    N_SH_gt[..., 3] = 1
    N_SH_gt[..., 4] = n[..., 0] * n[..., 1]
    N_SH_gt[..., 5] = n[..., 0] * n[..., 2]
    N_SH_gt[..., 6] = n[..., 2] * n[..., 1]
    N_SH_gt[..., 7] = n[..., 0] ** 2 - n[..., 1] ** 2
    N_SH_gt[..., 8] = 3 * n[..., 2] ** 2 - 1
    return N_SH_gt


def mae_map(n, gt_n):
    d = np.sum(np.multiply(n, gt_n), axis=-1)
    d[d > 1] = 1.0
    return np.rad2deg(np.arccos(d))


def reprojection_RMSE(I1, I2, mask):
    return np.mean(np.sqrt((I1[mask] - I2[mask]) ** 2))


def generate_rho_sphere(d=500):
    center_x = int(d / 2)
    center_y = int(d / 2)
    r = d / 2
    II, JJ = np.meshgrid(range(d), range(d))
    mask = r ** 2 - (II - center_x) ** 2 - (JJ - center_y) ** 2 < 1e-12

    albedo = np.zeros((d, d, 3))
    albedo[0: center_x, 0:center_y] = [1, 0.1, 0.1]  # red
    albedo[0: center_x, center_y:] = [0.1, 1, 0.1]  # green
    albedo[center_x:, 0:center_y] = [1, 1, 0.1]  # yellow
    albedo[center_x:, center_y:] = [0.1, 0.1, 1]  # blue
    albedo[mask] = [0, 0, 0]
    return albedo


def get_sparse_matrix_operator(mask):
    imask = np.flatnonzero(mask)
    I = []  # vertical index
    J = []  # horizontal index
    for i in imask:
        I.append(i // mask.shape[1])  # quotient
        J.append(i % mask.shape[1])  # remaining

    sub2ind = {(x, J[idx]): idx for idx, x in enumerate(I)}
    ind2sub = {value: key for key, value in sub2ind.items()}
    len_D = len(I)

    data_x = []
    row_x = []
    col_x = []
    for i in tqdm(range(len_D)):
        # i is the index for the flattened depth map
        # the pixel has left and right neighbour
        self_x = ind2sub[i][0]
        self_y = ind2sub[i][1]
        left = self_y - 1
        right = self_y + 1

        # When there is a neighbor on the right and the left: finite difference
        if (self_x, left) in sub2ind and (self_x, right) in sub2ind:
            row_x.append(i)
            row_x.append(i)
            data_x.append(-0.5)
            data_x.append(0.5)
            col_x.append(sub2ind[(self_x, left)])
            col_x.append(sub2ind[(self_x, right)])

        # the pixel has no left but right neighbour. [-1_, 1]
        if (self_x, left) not in sub2ind and (self_x, right) in sub2ind:
            row_x.append(i)
            row_x.append(i)
            data_x.append(-1)
            data_x.append(1)
            col_x.append(sub2ind[(self_x, self_y)])
            col_x.append(sub2ind[(self_x, right)])

        # the pixel has no right but left neighbour. [-1, 1_]
        if (self_x, left) in sub2ind and (self_x, right) not in sub2ind:
            row_x.append(i)
            row_x.append(i)
            data_x.append(-1)
            data_x.append(1)
            col_x.append(sub2ind[(self_x, left)])
            col_x.append(sub2ind[(self_x, self_y)])

    D_horizontal = coo_matrix((data_x, (row_x, col_x)), shape=(len_D, len_D)).tocsr()

    data_y = []
    row_y = []
    col_y = []
    for i in tqdm(range(len_D)):
        # i is the index for the flattened depth map
        self_x = ind2sub[i][0]
        self_y = ind2sub[i][1]
        up = self_x - 1
        bottom = self_x + 1

        # the pixel has bottom and up neighbour:
        if (up, self_y) in sub2ind and (bottom, self_y) in sub2ind:
            row_y.append(i)
            row_y.append(i)
            data_y.append(-0.5)
            data_y.append(0.5)
            col_y.append(sub2ind[(up, self_y)])
            col_y.append(sub2ind[(bottom, self_y)])

        # the pixel has no upper but bottom neighbour. [-1_, 1]^T
        if (up, self_y) not in sub2ind and (bottom, self_y) in sub2ind:
            row_y.append(i)
            row_y.append(i)
            data_y.append(-1)
            data_y.append(1)
            col_y.append(sub2ind[(self_x, self_y)])
            col_y.append(sub2ind[(bottom, self_y)])

        # the pixel has no bottom neighbour. [-1, 1_]^T
        if (up, self_y) in sub2ind and (bottom, self_y) not in sub2ind:
            row_y.append(i)
            row_y.append(i)
            data_y.append(-1)
            data_y.append(1)
            col_y.append(sub2ind[(up, self_y)])
            col_y.append(sub2ind[(self_x, self_y)])

    D_vertical = coo_matrix((data_y, (row_y, col_y)), shape=(len_D, len_D)).tocsr()
    return D_horizontal, D_vertical, ind2sub


def generate_laplacian_sparse_matrix(mask):
    """

    :param mask: a 2D binary mask
    :return:
    """
    mask_f = mask.flatten()
    H, W = mask.shape
    off_set = [-W, -1, 0, 1, W]
    diagnols = [-0.125, -0.125, 0.5, -0.125, -0.125]
    laplacian_full = sparse.diags(diagnols, off_set, shape=(H * W, H * W), format="csr")
    laplacian = laplacian_full[np.where(mask_f)].transpose()
    laplacian = laplacian[np.where(mask_f)].transpose()
    laplacian_err = np.squeeze(np.array(laplacian.sum(axis=1)))
    diag_err = scipy.sparse.diags(laplacian_err, format="csr")
    laplacian -= diag_err
    return laplacian


def laplacian_filtered(D_l, i, mask):
    H, W = i.shape
    # apply laplacian filter in a vectorized way
    s = D_l * i.flatten()[mask.flatten()]
    # recover the vector to matrix
    z_f = np.zeros_like(i).flatten()
    z_f[mask.flatten()] = s
    z_f = z_f.reshape((H, W))
    z_f[~mask] = np.NaN
    return z_f


def get_normal_from_z(z, mask, sx=1, sy=1):
    """
    compute the normal map from depth map by sparse matrix cmultiplication.
    n_object is only for rendering normal map and it's mapped to (0, 1).
    :param z:
    :param mask:
    :return:
    """
    H, W = mask.shape
    imask = np.flatnonzero(mask)
    z_flaten = z.ravel()[imask]
    D_h, D_v, ind2sub = get_sparse_matrix_operator(mask)
    flatten_z = coo_matrix(np.squeeze(z_flaten))
    zx = -np.squeeze(np.array((D_v * (flatten_z.tocsr().transpose())).todense())) / sx
    zy = np.squeeze(np.array((D_h * (flatten_z.tocsr().transpose())).todense())) / sy
    n_world = recover_normal_map(zx, zy, ind2sub, H, W)
    n_object = world_to_object(n_world.copy())

    n_world[~mask] = [0, 0, 0]
    n_object[~mask] = [0, 0, 0]
    n_object = (n_object + 1) / 2
    return n_world, n_object


def get_normal_from_z_perspective(z, mask, K, ratio=2):
    """
    compute the normal map from depth map by sparse matrix cmultiplication.
    n_object is only for rendering normal map and it's mapped to (0, 1).
    :param z:
    :param mask:
    :return:  D_1: fx * D_x, D_2: fy * D_y, D_3: - D_x * x - D_y * y - I
    """
    x0_in_world = K[0, 2] / ratio
    y0_in_world = K[1, 2] / ratio
    fx = K[0, 0]
    fy = K[1, 1]
    H, W = mask.shape
    imask = np.flatnonzero(mask)
    yy, xx = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
    xx, yy = xx.astype(np.float64), yy.astype(np.float64)
    xx = xx[::-1, :]
    xx -= (H - y0_in_world)
    yy -= (W - x0_in_world)
    xx_flatten = xx.ravel()[imask][:, None]
    yy_flatten = yy.ravel()[imask][:, None]

    z_flaten = z.ravel()[imask]
    D_h, D_v, ind2sub = get_sparse_matrix_operator(mask)
    D_x, D_y = - D_v, D_h
    D_1, D_2 = fx * D_x / ratio, fy * D_y / ratio
    D_3 = - D_x.multiply(xx_flatten) - D_y.multiply(yy_flatten) - sparse.identity(len(imask)).tocsr()

    flatten_z = coo_matrix(np.squeeze(z_flaten))
    zx = np.squeeze(np.array((D_1 * (flatten_z.tocsr().transpose())).todense()))
    zy = np.squeeze(np.array((D_2 * (flatten_z.tocsr().transpose())).todense()))
    zz = np.squeeze(np.array((D_3 * (flatten_z.tocsr().transpose())).todense()))

    theta = np.concatenate((zx[:, None], zy[:, None], zz[:, None]), axis=1)
    theta = normalize(theta, axis=1)

    n = np.zeros((mask.size, 3), dtype=np.float64)
    n[imask] = theta

    n_world = n.reshape((H, W, 3))
    n_object = world_to_object(n_world.copy())

    n_world[~mask] = [0, 0, 0]
    n_object[~mask] = [0, 0, 0]
    n_object = (n_object + 1) / 2
    return n_world, n_object, D_1, D_2, D_3


def histogram_equlization(img, level=256):
    if img.ndim == 2:
        H, W = img.shape
        PMF, bins = np.histogram(img, np.arange(level), normed=True)
        CDF = np.add.accumulate(PMF)
        new_gray_lvl = np.floor(CDF * (level - 1))
        new_img = new_gray_lvl[img.flatten()].reshape((H, W))
    else:
        new_img = np.zeros_like(img)
        for i in range(3):
            PMF, bins = np.histogram(img[..., i], np.arange(level), normed=True)
            CDF = np.add.accumulate(PMF)
            new_gray_lvl = np.floor(CDF * (level - 1))

            new_img[..., i] = new_gray_lvl[img[..., i].flatten()].reshape((H, W))
    return new_img


def prepare_intensity_image(n, mask, l, use_uniform_albedo, flash_x=0, flash_intensity=1):
    """
    render flash/no-flash image pair.
    m_e = rho * n_SH * l_e
    m_f = rho * n_SH * (l_e + l_f)

    :param n: normal map
    :param mask: object area
    :param use_uniform_albedo:  uniform or variant albedo
    :param flash_intensity: l_f = s * u
    :param lighting_type: envrionmental lighting type.
    :return:
    """
    H, W = mask.shape
    flash = flash_intensity * np.array([[flash_x, 0, -1, 0, 0, 0, 0, 0, 0]])

    if use_uniform_albedo:
        albedo = np.ones((H, W, 3))
    else:
        albedo = generate_rho_sphere()
    albedo[~mask] = [0, 0, 0]

    # render no-flash image
    m_e = render_SH(l, rho=albedo, n=n.copy(), obj_mask=mask)
    # render flash image
    m_f = render_SH(flash + l, rho=albedo, n=n.copy(), obj_mask=mask)

    # for visulization of lighting
    lighting = render_SH(l)

    return np.squeeze(m_e), np.squeeze(m_f), lighting


def reprojection_RMSE(I1, I2, mask):
    return np.mean(np.sqrt((I1[mask] - I2[mask]) ** 2))


def convert_vector_to_map(z, mask, background_value=np.NaN):
    z_map = np.zeros((mask.size,))
    mask_flatten = np.flatnonzero(mask)
    z_map[mask_flatten] = z
    z_map = z_map.reshape(mask.shape)
    z_map[~mask] = background_value
    return z_map


def crop_mask(mask):
    if mask.dtype is not np.uint8:
        mask = mask.astype(np.uint8) * 255
    im = Image.fromarray(mask)
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, 0)
    bbox = diff.getbbox()
    return bbox


def crop_image_by_mask(img, mask):
    bbox = crop_mask(mask)
    return img.copy()[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def sparse_matrix_multiplication(img, mask, operator):
    img_flatten = img.flatten()[mask.flatten()]
    img_flatten = coo_matrix(np.squeeze(img_flatten))
    result = np.array((operator * (img_flatten.tocsr().transpose())).todense())
    return result
