import sys
import time
from pprint import pprint

from scipy.optimize import minimize

from depth_normal_fusion import depth_normal_fusion
from draw_results import draw_results
from utils import *


def cost_function_and_jacobian(theta, n_intial, mr, l, imask, lambda1, lambda2, confidence):
    npix = len(imask)

    theta_1 = theta[:npix]
    theta_2 = theta[npix:(npix * 2)]
    theta_3 = theta[(npix * 2):]

    n1_initial = n_intial[:npix]
    n2_initial = n_intial[npix:(npix * 2)]
    n3_initial = n_intial[(npix * 2):]

    dz = np.sqrt(theta_1 ** 2 + theta_2 ** 2 + theta_3 ** 2)
    dz2 = dz ** 2

    N1 = theta_1 * dz
    N2 = theta_2 * dz
    N3 = theta_3 * dz
    N4 = dz2
    N5 = theta_1 * theta_2
    N6 = theta_1 * theta_3
    N7 = theta_2 * theta_3
    N8 = theta_1 ** 2 - theta_2 ** 2
    N9 = 3 * theta_3 ** 2 - dz ** 2

    N1_1 = dz + theta_1 ** 2 / dz
    N2_1 = theta_2 * theta_1 / dz
    N3_1 = theta_3 * theta_1 / dz
    N4_1 = 2 * theta_1
    N5_1 = theta_2
    N6_1 = theta_3
    N7_1 = 0
    N8_1 = 2 * theta_1
    N9_1 = - 2 * theta_1

    N1_2 = theta_2 * theta_1 / dz
    N2_2 = dz + theta_2 ** 2 / dz
    N3_2 = theta_2 * theta_3 / dz
    N4_2 = 2 * theta_2
    N5_2 = theta_1
    N6_2 = 0
    N7_2 = theta_3
    N8_2 = -2 * theta_2
    N9_2 = -2 * theta_2

    N1_3 = theta_3 * theta_1 / dz
    N2_3 = theta_2 * theta_3 / dz
    N3_3 = dz + theta_3 ** 2 / dz
    N4_3 = 2 * theta_3
    N5_3 = 0
    N6_3 = theta_1
    N7_3 = theta_2
    N8_3 = 0
    N9_3 = 4 * theta_3

    cost = 0
    jac = np.zeros(3 * npix)
    for ch in range(1):
        NL = l[ch, 0] * N1 + l[ch, 1] * N2 + l[ch, 2] * N3 + l[ch, 3] * N4 + l[ch, 4] * N5 + l[ch, 5] * N6 \
             + l[ch, 6] * N7 + l[ch, 7] * N8 + l[ch, 8] * N9

        delta_shading = NL + mr[:, ch] * N3
        cost += confidence * np.sum(delta_shading ** 2)

        DFD1 = l[ch, 0] * N1_1 + l[ch, 1] * N2_1 + l[ch, 2] * N3_1 + l[ch, 3] * N4_1 + l[ch, 4] * N5_1 \
               + l[ch, 5] * N6_1 + l[ch, 6] * N7_1 + l[ch, 7] * N8_1 + l[ch, 8] * N9_1 + mr[:, ch] * N3_1
        DFD2 = l[ch, 0] * N1_2 + l[ch, 1] * N2_2 + l[ch, 2] * N3_2 + l[ch, 3] * N4_2 + l[ch, 4] * N5_2 \
               + l[ch, 5] * N6_2 + l[ch, 6] * N7_2 + l[ch, 7] * N8_2 + l[ch, 8] * N9_2 + mr[:, ch] * N3_2
        DFD3 = l[ch, 0] * N1_3 + l[ch, 1] * N2_3 + l[ch, 2] * N3_3 + l[ch, 3] * N4_3 + l[ch, 4] * N5_3 \
               + l[ch, 5] * N6_3 + l[ch, 6] * N7_3 + l[ch, 7] * N8_3 + l[ch, 8] * N9_3 + mr[:, ch] * N3_3

        jac += np.concatenate((2 * confidence * DFD1 * delta_shading,
                               2 * confidence * DFD2 * delta_shading,
                               2 * confidence * DFD3 * delta_shading), axis=0)
    delta_initial = 1 - theta_1 * n1_initial - theta_2 * n2_initial - theta_3 * n3_initial
    delta_unit = 1 - theta_1 ** 2 - theta_2 ** 2 - theta_3 ** 2
    cost += lambda1 * np.sum(delta_initial ** 2) + lambda2 * np.sum(delta_unit ** 2)
    jac -= np.concatenate((2 * lambda1 * delta_initial * n1_initial,
                           2 * lambda1 * delta_initial * n2_initial,
                           2 * lambda1 * delta_initial * n3_initial), axis=0)
    jac -= np.concatenate((2 * lambda2 * delta_unit * theta_1,
                           2 * lambda2 * delta_unit * theta_2,
                           2 * lambda2 * delta_unit * theta_3), axis=0)
    return cost, jac


def estimate_lighting(m, n, mask, nb_harmonics=9):
    if m.ndim == 3:
        H, W, C = m.shape
    else:
        H, W = m.shape
        C = 1  # number of channels

    imask = np.flatnonzero(mask)

    N = normal_map_to_sh_map(n)
    N_flat = np.reshape(N, (H*W, 9))[imask]

    # Estimate lighting for each channel
    s = np.zeros((C, nb_harmonics))
    for ch in range(C):
        m_ch = m[..., ch]
        m_ch = m_ch.flatten()[imask]

        # remove nan value and extreme value in m_ch
        nan_mask = (np.isnan(m_ch) + (m_ch > 100) + (m_ch < -100)).astype(np.bool)
        m_ch = m_ch[~nan_mask]
        N_ch = N_flat[~nan_mask]

        # if one row of N contain nan, discard it.
        N_nan_mask = np.sum(np.isnan(N_ch), axis=-1).astype(np.bool)
        N_ch = N_ch[~N_nan_mask]
        m_ch = m_ch[~N_nan_mask]

        n_ = - N_ch / N_ch[..., 2:3]
        s[ch] = np.linalg.lstsq(n_, m_ch, rcond=None)[0]  # in this case, m_ch is m_e / (m_e - m_f)
    return s


def main(data, options):
    results = dict()
    results["no_flash_rgb"] = data["mnf_rgb"].copy()
    results["mask"] = data["mask"].copy()
    mask = data["mask"].copy()
    assert np.all(np.isclose(np.sum(data["n_init"] ** 2, -1)[mask], 1))
    m_r = data["mnf_rgb"] / (data["mf_rgb"] - data["mnf_rgb"])  # 1500 X 2048
    m_r[~mask] = np.NaN
    mask = np.logical_and(mask, ~np.isnan(np.sum(m_r, -1)))
    imask = np.flatnonzero(mask.copy())
    npix = len(imask)
    H, W, C = m_r.shape

    if options["use_confidence"]:
        mf = cv2.cvtColor(swap_RB_channel(data["mf_rgb"].astype(np.uint8)), cv2.COLOR_BGR2GRAY)
        mnf = cv2.cvtColor(swap_RB_channel(data["mnf_rgb"].astype(np.uint8)), cv2.COLOR_BGR2GRAY)
        ratio = mf / mnf
        ratio[~mask] = np.NaN
        mu = np.nanmean(ratio)
        sigma = np.nanstd(ratio)
        confidence_map = np.exp(-(ratio - mu) ** 2 / (2 * sigma ** 2))
        results["confidence_map"] = confidence_map
        confidence_map_flat = confidence_map.flatten()[imask]
    else:
        confidence_map_flat = np.ones_like(imask)

    mask_lighting = np.logical_and(~np.isclose(np.sum(data["n_init"] ** 2, -1), 0), mask)
    initial_lighting = estimate_lighting(m_r.copy(), data["n_init"].copy(), mask_lighting.copy())
    print("initial lighting is:\n{}".format(initial_lighting))
    initial_lighting_image = render_SH(initial_lighting)
    initial_shading = render_SH(initial_lighting, n=data["n_init"], obj_mask=mask)

    results["initial_lighting"] = initial_lighting_image
    results["initial_shading"] = initial_shading
    results["initial_albedo"] = data["mf_rgb"] / initial_shading

    I = np.reshape(m_r, (-1, 3))[imask]
    assert np.all(~np.isnan(I))
    _, _, ind2sub = get_sparse_matrix_operator(mask.copy())

    n1_initial = data["n_init"][..., 0].flatten()[imask]
    n2_initial = data["n_init"][..., 1].flatten()[imask]
    n3_initial = data["n_init"][..., 2].flatten()[imask]


    # initial noraml map
    N_initial_object = world_to_object(data["n_init"])
    N_initial_object[~mask] = [0, 0, 0]

    results["initial_normal"] = (N_initial_object + 1) / 2
    n_initial = np.squeeze(np.concatenate((n1_initial[:, None], n2_initial[:, None], n3_initial[:, None]), axis=0))

    # Nonlinear theta update
    n = minimize(cost_function_and_jacobian, x0=n_initial,
                     args=(
                     n_initial, I, initial_lighting, imask, options["lambda1"], options["lambda2"], confidence_map_flat),
                     jac=True,
                     options={"maxiter": options["maxit_bfgs"], "disp": 1, "ftol": options["tolX_bfgs"],
                              "gtol": options["tolFun_bfgs"]},
                     method="L-BFGS-B").x
    n1_estimated = n[:npix]
    n2_estimated = n[npix:(npix * 2)]
    n3_estimated = n[(npix * 2):]
    n_estimated_world = recover_normal_map(- n1_estimated/n3_estimated, - n2_estimated/n3_estimated, mask)

    # for normal map visualization
    n_estimated_object = world_to_object(n_estimated_world)
    n_estimated_object[~mask] = [0, 0, 0]
    n_estimated_object = (n_estimated_object + 1) / 2

    # depth normal fusion
    z_final_flat = depth_normal_fusion(data["z_init"], n, mask, data["K"], options["fusion_lamda"])
    estimated_z = convert_vector_to_map(z_final_flat, mask)

    # save results
    results["estimated_normal_object"] = n_estimated_object
    results["estimated_normal_world"] = n_estimated_world

    estimated_lighting = estimate_lighting(m_r.copy(), n_estimated_world.copy(), mask_lighting.copy())
    print("estimated lighting is:\n{}".format(estimated_lighting))
    results["estimated_lighting"] = render_SH(estimated_lighting)
    results["estimated_shading"] = render_SH(initial_lighting, n=n_estimated_world, obj_mask=mask)
    results["estimated_albedo"] = data["mf_rgb"] / results["estimated_shading"]

    with open(osp(options["fig_path"], "results.pickle"), "wb") as f:
        pickle.dump(results, f)
    draw_results(osp(options["fig_path"], "results.pickle"), options["fig_path"])

    estimated_points = data["pts"].copy()
    estimated_points[..., -1] = estimated_z
    estimated_points[~data["mask"]] = np.NaN
    sio.savemat(osp(options["fig_path"], "estimated_z.mat"), {"depth": estimated_points})
    return


if __name__ == "__main__":
    options = dict()
    options["maxit_bfgs"] = 200
    options["tolX_bfgs"] = 1e-17  # stopping criterion for the BFGS iterations
    options["tolFun_bfgs"] = 1e-17  # stopping criterion for the BFGS iterations
    options["fusion_lamda"] = 0.01
    options["lambda1"] = 0.1
    options["lambda2"] = 0.1
    options["use_confidence"] = True

    st_time = str(time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())))
    mkdir("results")
    save_dir = osp("results", st_time)
    mkdir(save_dir)
    data_root_dir = "data"
    for data_dir in os.listdir(data_root_dir):
        options["fig_path"] = osp(save_dir, data_dir)
        mkdir(options["fig_path"])
        sys.stdout = open(os.path.join(options["fig_path"], "log.txt"), "w")  # set log file
        data_dir = osp(data_root_dir, data_dir)
        data = data_loader(data_dir)
        sio.savemat(osp(options["fig_path"], "initial_z.mat"), {"depth": data["pts"]})

        pprint(options)
        main(data, options)
        sys.stdout.close()
