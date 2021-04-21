import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import cv2 as cv

from src.DirectionLight import drawUnitCircle
from src.utils import rescaleFrame

SAMPLING_PATH = "rsc/results/learn/sampling"
PREDICTIONS_PATH = "rsc/results/learn/predictions"


def interpolateSampling(xy, z, pixel, resolution=64, save_plot=False):
    if save_plot:
        plt.figure()
        plt.title('Learn Pixel #' + str(pixel))
        plt.scatter(x=xy[0, :], y=xy[1, :], c=z)
        plt.colorbar()
        plt.savefig(SAMPLING_PATH + "/" + str(pixel) + '.png')

    rbf = scipy.interpolate.Rbf(xy[0, :], xy[1, :], z, smooth=0.01, function='linear')
    XX, YY = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))

    z_interpolation = rbf(XX, YY)

    if save_plot:
        plt.figure()
        plt.title('Learn Pixel: (' + str(pixel[0]) + ',' + str(pixel[1]) + ')')
        plt.pcolormesh(XX, YY, z_interpolation, shading='flat', vmin=0, vmax=255)
        plt.colorbar()
        plt.scatter(x=xy[0, :], y=xy[1, :], c=z, vmin=0, vmax=255)
        plt.savefig(PREDICTIONS_PATH + "/" + str(pixel) + '.png')

    z_interpolation[z_interpolation > 255] = 255
    z_interpolation[z_interpolation < 0] = 0

    return z_interpolation


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global right_clicks
    if event == cv.EVENT_LBUTTONDOWN:
        right_clicks[0] = x
        right_clicks[1] = y


def creteImageRTI(tensor):
    _, _, n_x, n_y = tensor.shape
    img = np.zeros((n_y, n_x), np.uint8)
    img_circle = drawUnitCircle(n=64)
    while True:
        cv.namedWindow("RTI img")
        cv.setMouseCallback("RTI img", on_EVENT_LBUTTONDOWN)
        cv.imshow("RTI img", rescaleFrame(img_circle, 4))

        # wait for a key to be pressed to exit
        cv.waitKey(0)

        mouse_x, mouse_y = int(right_clicks[0] / 4), int(right_clicks[1] / 4)
        print("x: ", mouse_x, " y: ", mouse_y)

        for x in range(n_x):
            for y in range(n_y):
                img[y, x] = round(tensor[mouse_x, mouse_y, x, y])

        cv.imshow("RTI img", rescaleFrame(img, 4))
        # Listen to mouse events
        cv.waitKey(0)


right_clicks = [50, 50]
TENSOR = None
img = None
