import numpy as np

def square(rl, nr):
    x = np.linspace(-rl / 2, rl / 2, nr)
    mx, mp = np.meshgrid(x, x, indexing='ij')
    return mx, mp


def sphere(n_theta, n_phi):
    th = np.linspace(0, np.pi, n_theta)
    ph = np.linspace(0, 2*np.pi, n_phi)
    mth, mph = np.meshgrid(th, ph, indexing='ij')
    return mth, mph


def rectangle(xmin, xmax, ymin, ymax, nx, ny):
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    mx, my = np.meshgrid(x, y, indexing='ij')
    return mx, my
