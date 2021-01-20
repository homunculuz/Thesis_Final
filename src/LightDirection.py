import cv2 as cv
import numpy as np
from tqdm import tqdm
import os

from src.SamplingIntensity import PATH_POSES_DEBUG, getObjPoints, getTR
from src.utils import loadNP, saveMatrix

PATH_POSES_DEBUG = "rsc/results/debug_directory/poses"
PATH_SAMPLING_INTENSITY = "rsc/results/sampling/lc"
PATH_MATRIX_LIGHT_DIRECTION = "rsc/results/matrix/lc"


def getGradientPositions(ld_matrix, n=200):
    img = np.zeros((n, n), np.uint8)
    # draw localize
    val = round(n / 2)

    # unit circle with axis
    # cv.circle(img, (val, val), val, 255, 1)
    cv.line(img, (0, val), (2 * val, val), 255, 1)
    cv.line(img, (val, 0), (val, 2 * val), 255, 1)

    if len(ld_matrix) > 0:
        # project all x,y coordinates of illumination direction in the 2D plane
        for i in range(ld_matrix.shape[1]):
            x, y = val + ld_matrix[:, i] * 100
            # white color
            img[round(y), round(x)] = 255

    return img


def calculateLightDirectionMatrix(obj_grid, translation_vector, rotation_vector):
    n = obj_grid.shape[1]
    ld_m = np.zeros((3, n))
    rotation_matrix_t = cv.Rodrigues(rotation_vector)[0].transpose()
    ld_m = -rotation_matrix_t @ translation_vector - obj_grid
    ld_m = np.delete(ld_m, 2, 0)
    ld_m = ld_m / np.linalg.norm(ld_m, axis=-1)[:, np.newaxis]
    return ld_m


def getLightDirectionMatrix(path, x_resolution, y_resolution):
    # load parameters of the calibration
    k = loadNP("rsc/results/parameters/approximated_intrinsic_matrix.npy")
    dist = loadNP("rsc/results/parameters/approximated_parameters_distortion.npy")

    os.mkdir(PATH_POSES_DEBUG + '/' + os.path.basename(path))

    # get coordinate
    obj_grid = getObjPoints(x_resolution, y_resolution)

    for i in tqdm(range(len(os.listdir(path))), desc="Sampling Intensity"):
        ld_m = []
        frame = cv.imread(path + "/%d.jpg" % i)
        tr = getTR(frame, k, dist, i, path=PATH_POSES_DEBUG + '/' + os.path.basename(path))

        if len(tr) >= 0:
            ld_m = calculateLightDirectionMatrix(obj_grid, tr[0], tr[1])

        # save results
        cv.imwrite(PATH_SAMPLING_INTENSITY + "/%d.jpg" % i, getGradientPositions(ld_m))
        saveMatrix(PATH_MATRIX_LIGHT_DIRECTION, ld_m, i)

    return PATH_MATRIX_LIGHT_DIRECTION
