from tqdm import tqdm
import cv2 as cv
import numpy as np
from src.DirectionLight import getLightDirection, drawUnitCircle
from src.Interpolation import interpolateSampling
from src.SamplingIntensity import getObjPoints
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
    x = 100
    y = 249

    matrix_cc = []
    matrix_lc = []

    matrix_cc.extend(loadMatrix(path_cc))
    matrix_lc.extend(loadMatrix(path_lc))

    pixel_cc = np.zeros((1, len(matrix_cc)), dtype=np.float32)
    pixel_lc = np.zeros((2, len(matrix_lc)), dtype=np.float32)

    for pixel in tqdm(range(0, matrix_lc[0].shape[1], 1000), desc="Learning RTI"):
        img = drawUnitCircle()
        for n_frame in range(len(matrix_lc)):
            if matrix_lc[n_frame].size > 3:
                i_pixel = matrix_cc[n_frame].flatten()[pixel]
                new_img = getLightDirection(matrix_lc[n_frame][:, pixel], i_pixel)
                pixel_cc[:, n_frame] = i_pixel
                pixel_lc[:, n_frame] = matrix_lc[n_frame][:, pixel]
            img = cv.add(img, new_img)
        cv.imwrite(PATH_PIXELS + str(pixel) + ".jpg", img)
        interpolateSampling(pixel_lc[:, :201], pixel_cc[:, :201], pixel)

    frame = cv.imread("rsc/results/sampling/cc/0.jpg")
    frame[y, x] = 0
    cv.imshow("CC_LC", frame)
    cv.waitKey(100000)

    img = rescaleFrame(img, 4)
    cv.imshow("CC_LC", img)
    cv.waitKey(100000)

    cv.destroyAllWindows()
