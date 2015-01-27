from sklearn.datasets import make_swiss_roll
from sklearn import manifold

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def add_sub_plot(fig, index, data, t, proj=None):
    ax = fig.add_subplot(index, projection=proj)

    ax.scatter(*data.T, marker='o', c=t, cmap=plt.cm.Spectral)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    if proj is '3d':
        ax.set_zlabel('Z Label')

if __name__ == "__main__":
    X, t = make_swiss_roll(n_samples=1000)

    reconstruction_error = []
    for i in range(2,21):
        isomap = manifold.Isomap(i, 2)
        Y = isomap.fit_transform(X)
        reconstruction_error.append(isomap.reconstruction_error())

    #Find embedding with the lowest reconstruction error.
    best_fit = np.argmin(reconstruction_error)
    print "%d: %f" % (best_fit, reconstruction_error[best_fit])

    isomap = manifold.Isomap(best_fit, 2)
    Y = isomap.fit_transform(X)

    #plot reconstruction error rate over all trials
    plt.plot(reconstruction_error)
    plt.show()

    #Plot swiss roll manifold and lower dimensional mapping
    fig = plt.figure()
    add_sub_plot(fig, 121, X, t, proj='3d')
    add_sub_plot(fig, 122, Y, t)
    plt.show()
