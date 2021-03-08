from src.DirectionLight import getLightDirectionMatrix
from src.Interpolation import creteImageRTI
from src.ProjectIntenisityDirection import samplingPlotCCLC, pixelSpecific
from src.SamplingIntensity import getIntensityMatrix, getObjPoints
from src.Calibration import undistortedVideo
from src.Synchronize import callSynchronization
from src.utils import getFileNameFromPath, resetResults, resetLearn, loadNP, saveNP

PATH_CENTER_CAMERA = "rsc/videos/center_camera/cc1.mp4"
PATH_ILLUMINATION_CAMERA = "rsc/videos/light_camera/lc1.mp4"
PATH_UNDISTORTED_FRAMES = "rsc/results/undistorted_frames/"

X_RES = 640
Y_RES = 480


def getUndistortedFramesVideo(path_video, is_approximate, shift, num_skip=50):
    print("Getting frames: ", path_video)
    u_path_frames = PATH_UNDISTORTED_FRAMES + getFileNameFromPath(path_video)
    undistortedVideo(path_video, is_approximate=is_approximate, shift=shift, skip=num_skip)
    return u_path_frames


def getSynchroShift(path_cc, path_lc):
    path, shift = callSynchronization(path_cc, path_lc)
    shift_video_1 = 0
    shift_video_2 = 0
    if path == path_cc:
        shift_video_1 = shift
    else:
        shift_video_2 = shift
    return shift_video_1, shift_video_2


if __name__ == '__main__':
    """
    
    resetResults(reset_frames=True)

    shift_1, shift_2 = getSynchroShift(PATH_CENTER_CAMERA, PATH_ILLUMINATION_CAMERA)

    print(shift_1, shift_2)

    # get synchronized frames
    u_path_frames_cc = getUndistortedFramesVideo(path_video=PATH_CENTER_CAMERA, is_approximate=False, shift=shift_1,
                                                 num_skip=10)
    u_path_frames_lc = getUndistortedFramesVideo(path_video=PATH_ILLUMINATION_CAMERA, is_approximate=True,
                                                 shift=shift_2, num_skip=10)

    # create object-points mash grid
    obj_grid = getObjPoints(X_RES, Y_RES)

    path_im_cc = getIntensityMatrix("rsc/results/undistorted_frames/cc", obj_grid, x_resolution=X_RES,
                                    y_resolution=Y_RES)

    path_idm_lc = getLightDirectionMatrix("rsc/results/undistorted_frames/lc", obj_grid, x_resolution=X_RES,
                                          y_resolution=Y_RES)


    
    
    """

    resetLearn()
    tensor = pixelSpecific("rsc/results/matrix/cc/", "rsc/results/matrix/lc/")
    saveNP(tensor, "rsc/results/learn/tensor.npy")
    tensor = loadNP("rsc/results/learn/tensor.npy")
    creteImageRTI(tensor)
