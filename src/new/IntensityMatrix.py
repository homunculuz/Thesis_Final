from src.new.LightDirection import CharucoLightDirectionMatrix
import numpy as np
import cv2 as cv


class CharucoIntenstiyMatrix(CharucoLightDirectionMatrix):
    __descriptor = "Sampling Intensity"
    __path_k = "rsc/results/parameters/cc/intrinsic_matrix.npy"
    __path_dist = "rsc/results/parameters/cc/parameters_distortion.npy"
    __path_poses = "rsc/results/debug_directory/poses/cc"
    __path_sampling = "rsc/results/sampling/cc"
    __path_matrix = "rsc/results/matrix/cc"
    __X = 640
    __Y = 480

    def getImage(self, i_matrix):
        intensity_matrix = i_matrix.reshape((self.__X, self.__Y))
        img = np.zeros((self.__X, self.__Y), np.uint8)
        for i in range(self.__X):
            for j in range(self.__Y):
                img[i, j] = intensity_matrix[i, j]
        return img

    def calculateMatrix(self, translation_vector, rotation_vector, frame=None):
        n = self.obj_grid.shape[1]
        intensity_matrix = np.zeros((1, n))
        p = (self.k @ (cv.Rodrigues(rotation_vector)[0] @ self.obj_grid + translation_vector))
        p = p / p[2, :]
        print(p)
        # remove z coordinates and transformation
        p = np.delete(p, 2, 0).astype(int)
        for i in range(n):
            x, y = p[:, i]
            intensity_matrix[:, i] = frame[y, x][0]
        return intensity_matrix
