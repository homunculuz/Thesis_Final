import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

SAMPLING_PATH = "rsc/results/learn/sampling"
PREDICTIONS_PATH = "rsc/results/learn/predictions"


def interpolateSampling(xy, z, pixel):
    plt.figure()
    plt.title('Learn Pixel #' + str(pixel))
    plt.scatter(x=xy[0, :], y=xy[1, :], c=z)
    plt.colorbar()
    plt.savefig(SAMPLING_PATH + "/" + str(pixel) + '.png')

    rbf = scipy.interpolate.Rbf(xy[0, :], xy[1, :], z, smooth=0.01, function='linear')
    XX, YY = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))

    z_interpolation = rbf(XX, YY)

    plt.figure()
    plt.title('Learn Pixel #' + str(pixel))
    plt.pcolormesh(XX, YY, z_interpolation, shading='flat', vmin=0, vmax=255)
    plt.colorbar()
    plt.scatter(x=xy[0, :], y=xy[1, :], c=z, vmin=0, vmax=255)
    plt.savefig(PREDICTIONS_PATH + "/" + str(pixel) + '.png')

# interpolateSampling((np.random.rand(2, 10) - 0.5) * 2, np.random.rand(1, 10) * 255)
