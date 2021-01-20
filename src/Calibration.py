import numpy as np
import cv2 as cv
import os

from src.utils import saveNP, loadNP

PATH_CALIBRATION_VIDEO = "rsc/calibration_videos/"
PATH_UNDISTORTED_VIDEO = "rsc/results/undistorted_frames/"
PATH_PARAMETERS_CALIBRATION = "rsc/results/parameters/"

# parameters for calibration of the intensity_matrix
PATH_INTRINSIC_MATRIX = PATH_PARAMETERS_CALIBRATION + "intrinsic_matrix.npy"
PATH_DISTORTED_PAR = PATH_PARAMETERS_CALIBRATION + "parameters_distortion.npy"

PATH_APPROXIMATED_INTRINSIC_MATRIX = "rsc/results/parameters/approximated_intrinsic_matrix.npy"
PATH_APPROXIMATED_DISTORTED_PAR = "rsc/results/parameters/approximated_parameters_distortion.npy"

# DIM of black corners in the cheeseboard
X = 9
Y = 6


def reProjectionError(k, dist, r, t, obj_points, img_points):
    tot_error = 0
    for i in range(len(obj_points)):
        img_points_2, _ = cv.projectPoints(obj_points[i], r[i], t[i], k, dist)
        error = cv.norm(img_points[i], img_points_2, cv.NORM_L2) / len(img_points_2)
        tot_error += error
    return tot_error / len(obj_points)


def displayPoints(img, corners, ret, dim: tuple = (X, Y)):
    img = cv.drawChessboardCorners(img, dim, corners, ret)
    cv.imshow('img', img)
    cv.waitKey(500)


def undistortedVideo(path_video, is_approximate=True, skip=50, show_frames=False):
    # calculate intrinsic parameters
    if is_approximate:
        k, dist = approximatedCalibration(path_video)
    else:
        k, dist = calibrationFromChessboard()

    # open video
    cap = cv.VideoCapture(path_video)

    # width, height and the intensity per seconds of video
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Set up output video
    path = PATH_UNDISTORTED_VIDEO + os.path.basename(path_video)[:-4]
    # create directory
    os.mkdir(path)
    undistorted_frames_path = path + '/%d.jpg'

    i = 0
    count = -1

    while True:
        count = count + 1

        ret, frame = cap.read()
        if not ret:
            break

        if not count % skip == 0:
            continue

        new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(k, dist, (w, h), 1, (w, h))
        map_x, map_y = cv.initUndistortRectifyMap(k, dist, None, new_camera_mtx, (w, h), 5)
        dst = cv.remap(frame, map_x, map_y, cv.INTER_LINEAR)

        # grayscale
        dst = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)

        cv.imwrite(undistorted_frames_path % i, dst)
        i = i + 1

        if show_frames:
            cv.imshow('Undistorted Video Frames', dst)
            cv.waitKey(10)

    # print("Frames ", path, ": ", i)
    cap.release()
    cv.destroyAllWindows()
    return path


def calibration(is_display=False):
    print("Getting calibration....")
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_p = np.zeros((X * Y, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:X, 0:Y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.

    # open video
    file_name = PATH_CALIBRATION_VIDEO + os.listdir(PATH_CALIBRATION_VIDEO)[0]
    cap = cv.VideoCapture(file_name)

    # width, height and the intensity per seconds of video
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(frame_gray, (X, Y), None)

        # If found, add object points, image points (after refining them)
        if ret:
            obj_points.append(obj_p)
            corners2 = cv.cornerSubPix(frame_gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)
            if is_display:
                displayPoints(frame, corners2, ret)
        else:
            break
    cap.release()
    cv.destroyAllWindows()

    ret, k, dist, r, t = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)

    # save the computation
    saveNP(k, PATH_INTRINSIC_MATRIX)
    saveNP(dist, PATH_DISTORTED_PAR)

    return k, dist, r, t, img_points, obj_points


def calibrationFromChessboard():
    if not (os.path.isfile(PATH_INTRINSIC_MATRIX) and os.path.isfile(PATH_DISTORTED_PAR)):
        calibration()

    # load the saved the intrinsic parameters
    k = loadNP(PATH_INTRINSIC_MATRIX)
    dist = loadNP(PATH_DISTORTED_PAR)

    return k, dist


def approximatedCalibration(path):
    cap = cv.VideoCapture(path)

    # width, height and the intensity per seconds of video
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    k = [[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]]
    dist = np.zeros((4, 1))  # Assuming no lens distortion

    saveNP(k, PATH_APPROXIMATED_INTRINSIC_MATRIX)
    saveNP(dist, PATH_APPROXIMATED_DISTORTED_PAR)

    return np.array(k), np.array(dist)
