import os
import numpy as np
from tqdm import tqdm
import cv2 as cv
from src.SamplingIntensity import getTR
from src.utils import loadNP, saveMatrix
from sklearn.preprocessing import normalize

PATH_POSES_DEBUG = "rsc/results/debug_directory/poses/lc"
PATH_LIGHT_DIRECTION = "rsc/results/sampling/lc"
PATH_MATRIX_LIGHT_DIRECTION = "rsc/results/matrix/lc"


def calculateLightDirectionMatrix(rotation, translation, obj_grid):
    n = obj_grid.shape[1]
    ld_m = np.zeros((3, n))
    rotation_matrix_t = cv.Rodrigues(rotation)[0].transpose()
    ld_m = (-rotation_matrix_t) @ translation - obj_grid
    # normalized rows matrix
    ld_normalized_m = np.delete(ld_m, 1, 0) / ld_m.sum(axis=0)
    return ld_normalized_m


def getLightDirection(matrix, n=200):
    img = np.zeros((n, n), np.uint8)
    # draw localize
    val = round(n / 2)

    # unit circle with axis
    cv.circle(img, (val, val), val, 255, 1)
    cv.line(img, (0, val), (2 * val, val), 255, 1)
    cv.line(img, (val, 0), (val, 2 * val), 255, 1)

    if len(matrix) > 0:
        matrix = val + matrix
        # project all x,y coordinates of illumination direction in the 2D plane
        for i in range(matrix.shape[1]):
            x, y = matrix[:, i]
            # white color
            img[round(y), round(x)] = 255

    return img


def getLightDirectionMatrix(path, obj_grid, x_resolution, y_resolution):
    # load parameters of the calibration
    k = loadNP("rsc/results/parameters/lc/intrinsic_matrix.npy")
    dist = loadNP("rsc/results/parameters/lc/parameters_distortion.npy")

    for i in tqdm(range(len(os.listdir(path))), desc="Light Direction"):
        ld_m = []
        frame = cv.imread(path + "/%d.jpg" % i)
        tr = getTR(frame, k, dist, i, path=PATH_POSES_DEBUG)

        if len(tr) > 0:
            ld_m = calculateLightDirectionMatrix(tr[0], tr[1], obj_grid)

        # save results
        cv.imwrite(PATH_LIGHT_DIRECTION + "/%d.jpg" % i, getLightDirection(ld_m))
        saveMatrix(PATH_MATRIX_LIGHT_DIRECTION, ld_m, i)

    return PATH_LIGHT_DIRECTION
