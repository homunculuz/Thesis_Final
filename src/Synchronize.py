from scipy import signal
import numpy as np
from src.utils import blockPrint, enablePrint, sizeVideo2Seconds
from moviepy.editor import *
import soundfile as sf
import matplotlib.pyplot as plt

PATH_SOUNDS = "rsc/results/sounds/"


def plotWaves(sub_wave_matrix, wave_matrix, x_corr):
    plt.plot(sub_wave_matrix)
    plt.title("Sub-Wave Signal")
    plt.show()

    plt.plot(wave_matrix)
    plt.title("Wave Signal")
    plt.show()

    # normalize the cross correlation
    plt.plot(x_corr)
    plt.title("Cross-Correlation Plot")
    plt.show()


def findCutCorrelation(sub_wave_matrix, wave_matrix):
    sub_wave_matrix = sub_wave_matrix.flatten()
    wave_matrix = wave_matrix.flatten()
    x_corr = signal.correlate(sub_wave_matrix - np.mean(sub_wave_matrix), wave_matrix - np.mean(wave_matrix),
                              mode='valid') / (np.std(sub_wave_matrix) * np.std(wave_matrix) * len(wave_matrix))
    plotWaves(sub_wave_matrix, wave_matrix, x_corr)
    # getSound(sub_wave_matrix,44100)
    # getSound(wave_matrix,44100)
    return np.argmin(x_corr)


def Stereo2MonoWave(path):
    wave, fs = sf.read(path, dtype='float32')
    wave = np.delete(wave, 1, 1)
    return fs, wave


def getSound(wave, fs):
    import sounddevice as sd
    sd.play(wave, fs)
    status = sd.wait()  # Wait until file is done playing


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
