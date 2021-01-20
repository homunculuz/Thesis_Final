from tqdm import tqdm
import os

import cv2 as cv
import numpy as np

from src.utils import loadNP, saveMatrix

PATH_POSES_DEBUG = "rsc/results/debug_directory/poses/"
PATH_MATRIX_INTENSITY = "rsc/results/matrix/cc"
PATH_SAMPLING_INTENSITY = "rsc/results/sampling/cc"


def getGrid(resolution_x, resolution_y):
    i_coords, j_coords = np.meshgrid(np.linspace(4, 11, resolution_x), np.linspace(2, 7, resolution_y), indexing='ij')
    coordinate_grid = np.array([i_coords, j_coords])
    return coordinate_grid


def getObjPoints(resolution_x, resolution_y):
    grid = getGrid(resolution_x, resolution_y)
    n = grid.shape[1]
    m = grid.shape[2]
    i = 0
    obj_m = np.zeros((3, n * m))
    for x in range(n):
        for y in range(m):
            obj_m[0:2, i] = grid[:, x, y]
            i = i + 1
    return obj_m


def getTR(frame, k, dist, i, path):
    # generate 16x9 charuco board
    aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
    board = cv.aruco.CharucoBoard_create(16, 9, 1, .8, aruco_dict)
    aruco_params = cv.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

    cv.aruco.refineDetectedMarkers(frame, board, corners, ids, rejected_img_points)

    if len(corners) > 0:  # if there is at least one marker detected
        charucoretval, charucoCorners, charucoIds = cv.aruco.interpolateCornersCharuco(corners, ids, frame, board)

        retval, rvec, tvec = cv.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, k, dist, None,
                                                               None)  # pose estimation from a charuco board

        if retval:
            im_with_charuco_board = cv.aruco.drawAxis(frame.copy(), k, dist, rvec, tvec,
                                                      3)  # axis length 100 can be changed according to your requirement
            # cv.imwrite(path + "/%d.jpg" % i, im_with_charuco_board)
            return [rvec, tvec]

    return []


def getImageIntensity(intensity_matrix, row, column):
    intensity_matrix = intensity_matrix.reshape((row, column))
    img = np.zeros((row, column), np.uint8)
    for i in range(row):
        for j in range(column):
            img[i, j] = intensity_matrix[i, j]
    return img


def calculateIntensityMatrix(frame, rotation_vector, translation_vector, grid, k):
    n = grid.shape[1]
    intensity_matrix = np.zeros((1, n))
    p = (k @ (cv.Rodrigues(rotation_vector)[0] @ grid + translation_vector))
    p = p / p[2, :]
    # remove z coordinates and transformation
    p = np.delete(p, 2, 0).astype(int)
    for i in range(n):
        x, y = p[:, i]
        intensity_matrix[:, i] = frame[y, x][0]
    return intensity_matrix


def getIntensityMatrix(path, x_resolution, y_resolution):
    # load parameters of the calibration
    k = loadNP("rsc/results/parameters/intrinsic_matrix.npy")
    dist = loadNP("rsc/results/parameters/parameters_distortion.npy")

    os.mkdir(PATH_POSES_DEBUG + '/' + os.path.basename(path))

    # get coordinate
    obj_grid = getObjPoints(x_resolution, y_resolution)

    for i in tqdm(range(len(os.listdir(path))), desc="Sampling Intensity"):
        i_m = []
        frame = cv.imread(path + "/%d.jpg" % i)
        tr = getTR(frame, k, dist, i, path=PATH_POSES_DEBUG + '/' + os.path.basename(path))

        if len(tr) >= 0:
            i_m = calculateIntensityMatrix(frame, tr[0], tr[1], obj_grid, k)

        # save results
        cv.imwrite(PATH_SAMPLING_INTENSITY + "/%d.jpg" % i, getImageIntensity(i_m, x_resolution, y_resolution))
        saveMatrix(PATH_MATRIX_INTENSITY, i_m, i)

    return PATH_MATRIX_INTENSITY
