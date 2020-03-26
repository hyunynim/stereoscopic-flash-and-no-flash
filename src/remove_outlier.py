from skimage.filters.rank import median
from skimage.morphology import disk
import numpy as np


def outliner_removal(disparity):
    window_size = 31
    threshold = 3
    disp = median((disparity).astype(np.uint16), disk(window_size), mask=~np.isnan(disparity))
    diff = np.abs(disp.astype(np.float32) - disparity)
    disparity[diff>threshold] = np.NaN
    return disparity

