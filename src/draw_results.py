from utils import *

osp = os.path.join

def draw_results(file_path, img_path):
    file_dir = os.path.dirname(file_path)
    file_dir = os.path.dirname(file_dir)
    mkdir(img_path)
    with open(file_path, "rb") as f:
        results = pickle.load(f)
    print(results.keys())
    mask = results["mask"]
    for key, item in results.items():
        if "confidence_map" in key:
            cv2.imwrite(osp(img_path, "confidence_map.png"),
                        crop_image_by_mask((item * 255).astype(np.uint8), results["mask"]))
        if "initial_normal" in key or "final_normal" in key:
            item[~mask] = [1, 1, 1]
            cv2.imwrite(osp(img_path, key + ".png"), crop_image_by_mask(swap_RB_channel(item), results["mask"]) * 255)

        if "estimated_normal_object" in key:
            item[~mask] = [1, 1, 1]
            cv2.imwrite(osp(img_path, key + ".png"), crop_image_by_mask(swap_RB_channel(item), results["mask"]) * 255)

        if "no_flash_rgb" in key:
            item[~mask] = 255
            cv2.imwrite(osp(img_path, key + "_masked.png"), crop_image_by_mask(swap_RB_channel(item), results["mask"]))

        if "initial_lighting" in key or "estimated_lighting" in key:
            cv2.imwrite(osp(img_path, key + ".png"), item * (255 / np.nanmax(item)))

        if "initial_shading" in key or "estimated_shading" in key:
            # print(np.nanmax(item))
            item = item * (255 / np.nanmax(item))
            item[~mask] = 255
            cv2.imwrite(osp(img_path, key + ".png"), crop_image_by_mask(item, results["mask"]))

        if "initial_albedo" in key or "estimated_albedo" in key:
            n_mask = np.sum(item < 0, -1).astype(np.bool)
            item[n_mask] = 255
            item *= 4
            n_mask = np.sum(item > 255, -1).astype(np.bool)
            item[n_mask] = 255
            item = swap_RB_channel(item)
            item[~mask] = 255
            item = crop_image_by_mask(item, mask)
            cv2.imwrite(osp(img_path, key + ".png"), item.astype(np.uint8))
