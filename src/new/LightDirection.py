import cv2 as cv
import numpy as np
from tqdm import tqdm
import os

from src.utils import loadNP, saveMatrix


class CharucoLightDirectionMatrix:
    __n = 200
    __descriptor = "Sampling Light Direction"
    __path_k = "rsc/results/parameters/lc/intrinsic_matrix.npy"
    __path_dist = "rsc/results/parameters/lc/parameters_distortion.npy"
    __path_poses = "rsc/results/debug_directory/poses/lc"
    __path_sampling = "rsc/results/sampling/lc"
    __path_matrix = "rsc/results/matrix/lc"

    def __init__(self, path, obj_grid):
        # load parameters of the calibration
        self.k = loadNP(self.__path_k)
        self.dist = loadNP(self.__path_dist)
        # get coordinate
        self.obj_grid = obj_grid
        self.path = path

    def getImage(self, ld_matrix):
        img = np.zeros((self.__n, self.__n), np.uint8)
        # draw localize
        val = round(self.__n / 2)

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

    def calculateMatrix(self, translation_vector, rotation_vector, frame=None):
        n = self.obj_grid.shape[1]
        ld_m = np.zeros((3, n))
        rotation_matrix_t = cv.Rodrigues(rotation_vector)[0].transpose()
        ld_m = -rotation_matrix_t @ translation_vector - self.obj_grid
        ld_m = np.delete(ld_m, 2, 0)
        ld_m = ld_m / np.linalg.norm(ld_m, axis=-1)[:, np.newaxis]
        return ld_m

    def getTR(self, frame, i):
        # generate 16x9 charuco board
        aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
        board = cv.aruco.CharucoBoard_create(16, 9, 1, .8, aruco_dict)
        aruco_params = cv.aruco.DetectorParameters_create()

        corners, ids, rejected_img_points = cv.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

        cv.aruco.refineDetectedMarkers(frame, board, corners, ids, rejected_img_points)

        if len(corners) > 0:  # if there is at least one marker detected
            charucoretval, charucoCorners, charucoIds = cv.aruco.interpolateCornersCharuco(corners, ids, frame, board)

            retval, rvec, tvec = cv.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, self.k, self.dist,
                                                                   None,
                                                                   None)  # pose estimation from a charuco board

            if retval:
                im_with_charuco_board = cv.aruco.drawAxis(frame.copy(), self.k, self.dist, rvec, tvec, 3)

                cv.imwrite(self.__path_poses + "/%d.jpg" % i, im_with_charuco_board)
                return [rvec, tvec]

        return []

    def getMatrix(self):
        for i in tqdm(range(len(os.listdir(self.path))), desc=self.__descriptor):
            ld_m = []
            frame = cv.imread(self.path + "/%d.jpg" % i)
            tr = self.getTR(frame, i)

            if len(tr) > 0:
                ld_m = self.calculateMatrix(tr[0], tr[1], frame)

            # save results
            cv.imwrite(self.__path_sampling + "/%d.jpg" % i, self.getImage(ld_m))
            saveMatrix(self.__path_matrix, ld_m, i)

        return self.__path_matrix
