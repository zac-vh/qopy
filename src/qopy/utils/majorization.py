import numpy as np
from qopy.phase_space.measures import integrate_2d as wig_int


def lorenz_decreasing(w, only_positive=True):
    v = w.ravel()
    if only_positive:
        v = np.maximum(v, np.zeros(len(v)))
    return np.cumsum(np.sort(v)[::-1])


def lorenz_increasing(w, only_negative=True):
    v = w.ravel()
    if only_negative:
        v = np.minimum(v, np.zeros(len(v)))
    return np.cumsum(np.sort(v))


def lorenz_decreasing_2d(w, rl, only_positive=True):
    nr = len(w)
    dim = len(w.shape)
    return lorenz_decreasing(w, only_positive) * (rl / (nr - 1)) ** dim


def lorenz_increasing_2d(w, rl, only_negative=True):
    nr = len(w)
    dim = len(w.shape)
    return lorenz_increasing(w, only_negative) * (rl / (nr - 1)) ** dim


def decreasing_rearrangement_2d(w, only_positive=True):
    nr = len(w)
    wd = np.sort(w.ravel())[::-1]
    if only_positive:
        wd = np.maximum(np.zeros(len(wd)), wd)
    x = np.linspace(-1, 1, nr)
    mx, my = np.meshgrid(x, x)
    m = mx ** 2 + my ** 2
    ind = np.unravel_index(np.argsort(m, axis=None), m.shape)
    wr = np.zeros([nr, nr])
    wr[ind] = wd
    return wr


def increasing_rearrangement_2d(w, only_negative=True):
    nr = len(w)
    wd = np.sort(w.ravel())
    if only_negative:
        wd = np.minimum(np.zeros(len(wd)), wd)
    x = np.linspace(-1, 1, nr)
    mx, my = np.meshgrid(x, x)
    m = mx ** 2 + my ** 2
    ind = np.unravel_index(np.argsort(m, axis=None), m.shape)
    wr = np.zeros([nr, nr])
    wr[ind] = wd
    return wr


def level_function_differential(w, interval, rl):
    nr = len(w)
    n = len(interval)
    dxdp = (rl/(nr-1))**2
    flev = np.zeros(n)
    for i in range(n-1):
        wi = (w >= interval[i])*(w < interval[i+1])
        flev[i] = wig_int(wi, rl)*dxdp
    flev[-1] = wig_int(wi >= interval[-1], rl)*dxdp
    return flev


def level_function_greater(w, interval, rl):
    nr = len(w)
    n = len(interval)
    dxdp = (rl / (nr - 1)) ** 2
    flev = np.zeros(n)
    for i in range(n):
        wi = (w >= interval[i])
        flev[i] = wig_int(wi, rl) * dxdp
    return flev


def level_function_less(w, interval, rl):
    nr = len(w)
    n = len(interval)
    dxdp = (rl / (nr - 1)) ** 2
    flev = np.zeros(n)
    for i in range(n):
        wi = (w <= interval[i])
        flev[i] = wig_int(wi, rl) * dxdp
    return flev