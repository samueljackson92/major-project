from sklearn.datasets import make_swiss_roll
from sklearn import manifold

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_3D(X):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(*X.T, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def plot_2D(X):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(*X.T, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

    plt.show()

if __name__ == "__main__":
    X, t = make_swiss_roll(n_samples=1000)
    Y = manifold.Isomap(10, 2).fit_transform(X)
    plot_3D(X)
    plot_2D(Y)
