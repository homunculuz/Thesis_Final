import glob
import shutil
import sys
import os
import cv2 as cv
import numpy as np
from tqdm import tqdm

def resetResults(reset_frames):
    if reset_frames:
        # reset undistorted_frames
        removeDirectory("rsc/results/undistorted_frames")
        createDirectoryStructure("undistorted_frames")

        # reset parameters
        removeDirectory("rsc/results/parameters")
        createDirectoryStructure("parameters")

    # reset sampling
    removeDirectory("rsc/results/sampling")
    createDirectoryStructure("sampling")
    os.mkdir("rsc/results/sampling" + "/cc_lc")


    # reset matrix
    removeDirectory("rsc/results/matrix")
    createDirectoryStructure("matrix")

    # reset debug_directory/poses
    removeDirectory("rsc/results/debug_directory/poses")
    createDirectoryStructure("debug_directory/poses")


def createDirectoryStructure(name):
    os.mkdir("rsc/results/" + name)
    os.mkdir("rsc/results/" + name + "/cc")
    os.mkdir("rsc/results/" + name + "/lc")


def removeDirectory(path):
    shutil.rmtree(path)


def removeFiles(path):
    file_list = glob.glob(os.path.join(path, "*"))
    for f in file_list:
        os.remove(f)


def getFileNameFromPath(path):
    return os.path.basename(path)[:-4]


def saveMatrix(path, matrix, i):
    p = path + '/%d.npy'
    saveNP(matrix, p % i)


def loadMatrix(path,label):
    list_matrix = []
    for i in tqdm(range(len(os.listdir(path))), desc=label):
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
