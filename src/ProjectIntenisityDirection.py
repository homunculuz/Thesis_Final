from tqdm import tqdm
import cv2 as cv

from src.DirectionLight import getLightDirection
from src.utils import loadMatrix, rescaleFrame

PATH_CC_LC = "rsc/results/sampling/cc_lc"


def samplingPlotCCLC(path_cc, path_lc):
    matrix_cc = []
    matrix_lc = []

    matrix_cc.extend(loadMatrix(path_cc, label="Loading CC Matrix"))
    matrix_lc.extend(loadMatrix(path_lc, label="Getting LC Matrix"))

    for i in tqdm(range(len(matrix_lc)), desc="SAVING CC_LC"):
        if matrix_lc[i].size <= 3:
            continue
        img = getLightDirection(matrix_lc[i], 200, matrix_cc[i].flatten().tolist())
        img = rescaleFrame(img, 3)

        # save results
        cv.imwrite(PATH_CC_LC + "/%d.jpg" % i, img)



