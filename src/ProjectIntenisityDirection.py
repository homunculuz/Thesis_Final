from tqdm import tqdm
import cv2 as cv
import numpy as np
from src.DirectionLight import getLightDirection, drawUnitCircle
from src.Interpolation import interpolateSampling
import matplotlib.pyplot as plt

from src.utils import loadMatrix, rescaleFrame

PATH_CC_LC = "rsc/results/sampling/cc_lc"
PATH_PIXELS = "rsc/results/learn/pixels/"


def samplingPlotCCLC(path_cc, path_lc):
    matrix_cc = []
    matrix_lc = []

    matrix_cc.extend(loadMatrix(path_cc))
    matrix_lc.extend(loadMatrix(path_lc))

    for i in tqdm(range(len(matrix_lc)), desc="SAVING CC_LC"):
        if matrix_lc[i].size <= 3:
            img = drawUnitCircle()
        else:
            img = getLightDirection(matrix_lc[i], matrix_cc[i].flatten().tolist())

        img = rescaleFrame(img, 3)
        # save results
        cv.imwrite(PATH_CC_LC + "/%d.jpg" % i, img)


def pixelSpecific(path_cc, path_lc):
    x1 = 100
    x2 = 155
    y1 = 160
    y2 = 210

    frame = cv.imread("rsc/results/sampling/cc/0.jpg")
    for x in range(x1, x2):
        for y in range(y1, y2):
            frame[y, x] = 0
    cv.imwrite(PATH_PIXELS[:-7] + "0-frame.jpg", frame)
    cv.destroyAllWindows()

    matrix_cc = []
    matrix_lc = []

    matrix_cc.extend(loadMatrix(path_cc))
    matrix_lc.extend(loadMatrix(path_lc))

    interpolation_tensor = np.zeros((64, 64, x2 - x1, y2 - y1), dtype=np.float32)

    for x in tqdm(range(x1, x2), desc="Sampling"):
        for y in range(y1, y2):
            img = drawUnitCircle()
            pixel_cc = np.zeros((1, len(matrix_cc)), dtype=np.float32)
            pixel_lc = np.zeros((2, len(matrix_lc)), dtype=np.float32)
            for n_frame in range(len(matrix_lc)):
                if matrix_lc[n_frame].size > 3:
                    reshaped_matrix_lc = np.reshape(matrix_lc[n_frame], (2, 640, 480))
                    i_pixel = (matrix_cc[n_frame])[x, y]
                    lc_pixel = reshaped_matrix_lc[:, x, y]
                    pixel_cc[:, n_frame] = i_pixel
                    pixel_lc[:, n_frame] = lc_pixel
                    new_img = getLightDirection(lc_pixel, i_pixel)
                img = cv.add(img, new_img)
            # same numbers of elements
            pixel_cc = pixel_cc[:, :pixel_lc.shape[-1]]

            # remove bad indices
            good_indices = pixel_cc.flatten() != 0
            pixel_cc = pixel_cc[:, good_indices]
            pixel_lc = pixel_lc[:, good_indices]

            # plt.hist(pixel_cc.flatten())
            # plt.show()

            cv.imwrite(PATH_PIXELS + str([x, y]) + ".jpg", rescaleFrame(img, 4))
            interpolation_tensor[:, :, x % x1, y % y1] = interpolateSampling(pixel_lc[:, :201], pixel_cc[:, :201],
                                                                             [x, y])
    return interpolation_tensor
