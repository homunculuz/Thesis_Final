import glob
import shutil
import sys
import os
import cv2 as cv
import numpy as np


def resetResults():
    # removeDirectory("rsc/results/undistorted_frames")
    # os.mkdir("rsc/results/undistorted_frames")
    removeDirectory("rsc/results/debug_directory/poses")
    os.mkdir("rsc/results/debug_directory/poses")


def removeDirectory(path):
    shutil.rmtree(path)


def removeFiles(path):
    file_list = glob.glob(os.path.join(path, "*"))
    for f in file_list:
        os.remove(f)


def getFileNameFromPath(path):
    return os.path.basename(path)[:-4]


def saveMatrix(list_matrix, path):
    p = path + '/%d.npy'
    for i, l in enumerate(list_matrix):
        saveNP(l, p % i)


def loadMatrix(path):
    list_matrix = []
    for i in range(len(os.listdir(path))):
        matrix_path = (path + "/%d.npy") % i
        list_matrix.append(loadNP(matrix_path))
    return list_matrix


def saveNP(data, path):
    with open(path, "wb") as f:
        np.save(f, data)


def loadNP(path):
    with open(path, "rb") as f:
        t = np.load(f)
    return np.array(t)


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def rescaleFrame(frame, percent=.60):
    width = int(frame.shape[1] * percent)
    height = int(frame.shape[0] * percent)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)


def sizeVideo2Seconds(path):
    cap = cv.VideoCapture(path)
    clip_size = int(cap.get(cv.CAP_PROP_FRAME_COUNT) / (cap.get(cv.CAP_PROP_FPS)))
    cap.release()
    return clip_size
