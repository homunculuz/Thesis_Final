import scipy.io.wavfile as wf
from scipy import signal
import numpy as np
from src.utils import blockPrint, enablePrint, sizeVideo2Seconds
from moviepy.editor import *

PATH_SOUNDS = "rsc/results/sounds/"


def findCutCorrelation(test, template):
    corr = signal.correlate(test, template, mode='valid')
    return np.argmax(corr)


def Stereo2MonoWave(path):
    fs, wave = wf.read(path)
    wave = np.delete(wave, 1, 1).astype(float)
    return fs, wave


def synchronizeVideo(path_video_1, path_video_2):
    blockPrint()
    # Extract the video1 audio, which is the audio of the reference clip
    video = VideoFileClip(path_video_1)
    audio = video.audio
    audio.write_audiofile(PATH_SOUNDS + "video1.wav")

    # Extract the video2 audio, which is the audio of the reference clip
    video = VideoFileClip(path_video_2)
    audio = video.audio
    audio.write_audiofile(PATH_SOUNDS + "video2.wav")
    enablePrint()

    # Obtains the video2 shift
    fs1, wave1 = Stereo2MonoWave(PATH_SOUNDS + "video1.wav")
    fs2, wave2 = Stereo2MonoWave(PATH_SOUNDS + "video2.wav")

    size_video = round(wave1.size / fs1, 2)
    # possible to round the result
    clip_start = round(findCutCorrelation(wave2, wave1) / fs1, 2)

    return clip_start


def callSynchronization(path_video_1, path_video_2):
    # get the size of video
    clip_size_1 = sizeVideo2Seconds(path_video_1)
    clip_size_2 = sizeVideo2Seconds(path_video_2)

    path_1 = path_video_1
    path_2 = path_video_2

    if clip_size_2 > clip_size_1:
        path_1 = path_video_2
        path_2 = path_video_1

    return path_1, synchronizeVideo(path_1, path_2)
