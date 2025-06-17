import numpy as np
import math

def grid_square(rl, nr):
    x = np.linspace(-rl / 2, rl / 2, nr)
    mx, mp = np.meshgrid(x, x, indexing='ij')
    return mx, mp


def grid_sphere(n_theta, n_phi):
    th = np.linspace(0, math.pi, n_theta)
    ph = np.linspace(0, 2*math.pi, n_phi)
    mth, mph = np.meshgrid(th, ph, indexing='ij')
    return mth, mph