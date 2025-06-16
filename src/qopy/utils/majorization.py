import numpy as np
from qopy.phase_space.measures import integrate_2d as wig_int


def lorenz_decreasing(w, only_positive=True):
    # Compute the Lorenz decreasing curve of any type of distribution
    # The input array is converted to a vector
    v = np.ravel(w)
    if only_positive:
        v = np.clip(v, 0, None)
    return np.concatenate(([0.0], np.cumsum(np.sort(v)[::-1])))


def lorenz_increasing(w, only_negative=True):
    # Compute the Lorenz increasing curve of any type of distribution
    # The input array is converted to a vector
    v = np.ravel(w)
    if only_negative:
        v = np.clip(v, None, 0)
    return np.concatenate(([0.0], np.cumsum(np.sort(v))))


def lorenz_decreasing_2d(w, rl, only_positive=True):
    # Compute the Lorenz decreasing curve of a (possibly multimode) phase-space function
    nr = len(w)
    dim = len(w.shape)
    return lorenz_decreasing(w, only_positive) * (rl / (nr - 1)) ** dim


def lorenz_increasing_2d(w, rl, only_negative=True):
    # Compute the Lorenz increasing curve of a (possibly multimode) phase-space function
    nr = len(w)
    dim = len(w.shape)
    return lorenz_increasing(w, only_negative) * (rl / (nr - 1)) ** dim


def spiral(vec, nr):
    # Arrange the vector vec into a spiral over a nr x nr grid
    x = np.linspace(-1, 1, nr)
    mx, my = np.meshgrid(x, x)
    m = mx ** 2 + my ** 2
    ind = np.unravel_index(np.argsort(m, axis=None), m.shape)
    wr = np.zeros([nr, nr])
    wr[ind] = vec
    return wr


def decreasing_rearrangement_2d(w, only_positive=False):
    # Compute the decreasing rearrangement of a single-mode phase-space function
    nr = len(w)
    wd = np.sort(np.ravel(w))[::-1]
    if only_positive:
        wd = np.clip(wd, 0, None)
    return spiral(wd, nr)


def increasing_rearrangement_2d(w, only_negative=False):
    # Compute the increasing rearrangement of a single-mode phase-space function
    nr = len(w)
    wd = np.sort(np.ravel(w))
    if only_negative:
        wd = np.clip(wd, None, 0)
    return spiral(wd, nr)


def relative_lorenz_decreasing(p, q):
    # Compute the relative Lorenz decreasing curve of any type of distributions
    # The input arrays are converted to vectors
    p, q = np.ravel(p), np.ravel(q)
    idx = np.argsort(-p/q)
    p_sorted, q_sorted = p[idx], q[idx]
    p_cum = np.concatenate(([0.0], np.cumsum(p_sorted)))
    q_cum = np.concatenate(([0.0], np.cumsum(q_sorted)))
    return p_cum, q_cum


def relative_lorenz_increasing(p, q):
    # Compute the relative Lorenz increasing curve of any type of distribution
    # The input array is converted to a vector
    p, q = np.ravel(p), np.ravel(q)
    idx = np.argsort(p/q)
    p_sorted, q_sorted = p[idx], q[idx]
    p_cum = np.concatenate(([0.0], np.cumsum(p_sorted)))
    q_cum = np.concatenate(([0.0], np.cumsum(q_sorted)))
    return p_cum, q_cum


def relative_lorenz_decreasing_2d(w, wref, rl):
    # Compute the Lorenz decreasing curve of a (possibly multimode) phase-space function
    nr = len(w)
    dim = len(w.shape)
    dx = (rl / (nr - 1))
    vol = dx**dim
    p, q = relative_lorenz_decreasing(w, wref)
    return p * vol, q * vol


def relative_lorenz_increasing_2d(w, wref, rl):
    # Compute the Lorenz decreasing curve of a (possibly multimode) phase-space function
    nr = len(w)
    dim = len(w.shape)
    dx = (rl / (nr - 1))
    vol = dx**dim
    p, q = relative_lorenz_increasing(w, wref)
    return p * vol, q * vol


def relative_decreasing_rearrangement(w, wref):
    nr = len(w)
    v1, v2 = np.ravel(w), np.ravel(wref)
    idx = np.argsort(-v1 / v2)
    v1_sorted, v2_sorted = v1[idx], v2[idx]
    return spiral(v1_sorted, nr), spiral(v2_sorted, nr)


def relative_increasing_rearrangement(w, wref):
    nr = len(w)
    v1, v2 = np.ravel(w), np.ravel(wref)
    idx = np.argsort(v1 / v2)
    v1_sorted, v2_sorted = v1[idx], v2[idx]
    return spiral(v1_sorted, nr), spiral(v2_sorted, nr)
