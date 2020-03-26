from scipy import sparse
import numpy as np
import scipy
import scipy.sparse.linalg


def generate_laplacian_sparse_matrix(mask, ratio=1):
    """

    :param mask: a binary mask.
    :return:
    """
    mask_f = mask.flatten()
    H, W = mask.shape
    off_set = [-W, -1, 0, 1, W]
    diagnols = np.array([-1, -1, 4, -1, -1], dtype=np.float64) / ratio   # /8
    laplacian_full = sparse.diags(diagnols, off_set, shape=(H*W, H*W), format="csr")
    # delete row and columns outside mask
    # note that csr format support only row indexing,
    # so we delete rows, transpose it, delete rows (columns in the origin matrix), and transpose it back.
    laplacian = laplacian_full[np.where(mask_f)].transpose()
    laplacian_op = laplacian[np.where(mask_f)].transpose()
    return laplacian_op


def construct_b_vector(img, mask):
    """
    img is a 2-d ndarray. mask is a integer ndarray w/ the same shape as img
    :param img:
    :param mask:
    :return:
    """
    assert img.ndim == 2
    assert img.shape == mask.shape
    if mask.dtype is np.bool:
        mask_bool = mask.copy()
        mask = mask.astype(np.int)
    else:
        mask_bool = mask.astype(np.bool)

    b = np.zeros_like(np.flatnonzero(mask), dtype=np.float64)
    # select vertical neighbour
    vertical_neighbour_index_matrix = mask[1:, :] - mask[:-1, :]
    up_padded_vertical = np.pad(vertical_neighbour_index_matrix, [(1, 0), (0, 0)], "constant")
    down_padded_vertical = np.pad(vertical_neighbour_index_matrix, [(0, 1), (0, 0)], "constant")
    b[(up_padded_vertical == 1).flatten()[mask_bool.flatten()]] += img[np.where(down_padded_vertical == 1)]
    b[(down_padded_vertical == -1).flatten()[mask_bool.flatten()]] += img[np.where(up_padded_vertical == -1)]
    # select horizontal neighbours
    hori_neighbour_index_matrix = mask[:, 1:] - mask[:, :-1]
    left_padded_vertical = np.pad(hori_neighbour_index_matrix, [(0, 0), (1, 0)], "constant")
    right_padded_vertical = np.pad(hori_neighbour_index_matrix, [(0, 0), (0, 1)], "constant")
    b[(left_padded_vertical == 1).flatten()[mask_bool.flatten()]] += img[np.where(right_padded_vertical == 1)]
    b[(right_padded_vertical == -1).flatten()[mask_bool.flatten()]] += img[np.where(left_padded_vertical == -1)]
    return b


def laplacian_solver_for_Dirichlet_boundary_condition(img, mask):
    """

    :param img:
    :param mask:
    :return:
    """
    img_temp = img.copy()  # avoid directly modifying the argument because ravel() creates a view
    A = generate_laplacian_sparse_matrix(mask)
    b = construct_b_vector(img_temp, mask.astype(np.int))

    x = scipy.sparse.linalg.spsolve(A, b)
    img_temp.ravel()[mask.flatten()] = x
    return img_temp


if __name__ == "__main__":
    # test code
    img = np.array([[1, 2, 3, 4],
                    [5, np.NaN, np.NaN, 8],
                    [9, np.NaN, np.NaN, 12],
                    [13, 14, 15, 16]], dtype=np.float64)
    mask = np.array([[0, 0, 0, 0],
                     [0, 1, 1, 0],
                     [0, 1, 1, 0],
                     [0, 0, 0, 0]], dtype=np.bool)
    img_filled = laplacian_solver_for_Dirichlet_boundary_condition(img, mask)
    print(img_filled)


