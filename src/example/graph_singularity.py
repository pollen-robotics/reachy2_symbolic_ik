import matplotlib.pyplot as plt
import numpy as np


def make_graph() -> None:
    """make a 3D graph of the shoulder pitch joint depending on the x and y position of the elbow on the shoulder frame"""
    x = np.linspace(-0.1, 0.1, 500)
    y = np.linspace(-0.1, 0.1, 500)
    x, y = np.meshgrid(x, y)
    z = -np.arctan2(y, x) % (2 * np.pi)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("-atan2(y, x)")
    plt.show()


def main_test() -> None:
    make_graph()


if __name__ == "__main__":
    main_test()
