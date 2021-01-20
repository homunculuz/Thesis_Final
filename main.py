import os

from src.LightDirection import getLightDirectionMatrix
from src.SamplingIntensity import getIntensityMatrix
from src.Calibration import undistortedVideo
from src.utils import resetResults, getFileNameFromPath

PATH_CENTER_CAMERA = "rsc/videos/center_camera/cc1.mp4"
PATH_ILLUMINATION_CAMERA = "rsc/videos/light_camera/lc1.mp4"

PATH_UNDISTORTED_FRAMES = "rsc/results/undistorted_frames/"

X_RES = 640
Y_RES = 480


def getUndistortedFramesVideo(path_video, is_approximate, num_skip=10):
    u_path_frames = PATH_UNDISTORTED_FRAMES + getFileNameFromPath(path_video)
    undistortedVideo(path_video, is_approximate=is_approximate, skip=num_skip)
    return u_path_frames


if __name__ == '__main__':
    resetResults()
    # u_path_frames_cc = getUndistortedFramesVideo(path_video=PATH_CENTER_CAMERA, is_approximate=False)
    # u_path_frames_lc = getUndistortedFramesVideo(path_video=PATH_ILLUMINATION_CAMERA, is_approximate=True)
    # path_im_cc = getIntensityMatrix("rsc/results/undistorted_frames/cc", x_resolution=X_RES, y_resolution=Y_RES)
    path_idm_lc = getLightDirectionMatrix("rsc/results/undistorted_frames/lc", x_resolution=X_RES, y_resolution=Y_RES)
