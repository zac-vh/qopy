import numpy as np
from qopy.phase_space.measures import integrate_2d as wig_int


def lorenz_decreasing(w, only_positive=True):
    # Compute the Lorenz decreasing curve of any type of distribution
    # The input array is converted to a vector
    v = w.ravel()
    if only_positive:
        v = np.maximum(v, np.zeros(len(v)))
    return np.cumsum(np.sort(v)[::-1])


def lorenz_increasing(w, only_negative=True):
    # Compute the Lorenz increasing curve of any type of distribution
    # The input array is converted to a vector
    v = w.ravel()
    if only_negative:
        v = np.minimum(v, np.zeros(len(v)))
    return np.cumsum(np.sort(v))


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


def decreasing_rearrangement_2d(w, only_positive=True):
    # Compute the decreasing rearrangement of a single-mode phase-space function
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
    # Compute the increasing rearrangement of a single-mode phase-space function
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


def relative_lorenz_decreasing(p, q):
    # Compute the relative Lorenz decreasing curve of any type of distributions
    # The input arrays are converted to vectors
    p = p.ravel()
    q = q.ravel()
    idx = np.argsort(p/q)[::-1]
    p_sorted, q_sorted = p[idx], q[idx]
    p_cum = np.concatenate(([0.0], np.cumsum(p_sorted)))
    q_cum = np.concatenate(([0.0], np.cumsum(q_sorted)))
    return p_cum, q_cum


def relative_lorenz_increasing(p, q):
    # Compute the Lorenz decreasing curve of any type of distribution
    # The input array is converted to a vector
    p = p.ravel()
    q = q.ravel()
    idx = np.argsort(p/q)
    p_sorted, q_sorted = p[idx], q[idx]
    p_cum = np.concatenate(([0.0], np.cumsum(p_sorted)))
    q_cum = np.concatenate(([0.0], np.cumsum(q_sorted)))
    return p_cum, q_cum


def relative_lorenz_decreasing_2d(w1, w2, rl):
    # Compute the Lorenz decreasing curve of a (possibly multimode) phase-space function
    nr = len(w1)
    dim = len(w1.shape)
    dx = (rl / (nr - 1))
    vol = dx**dim
    p, q = relative_lorenz_decreasing(w1, w2)
    return p * vol, q * vol


def relative_lorenz_increasing_2d(w1, w2, rl):
    # Compute the Lorenz decreasing curve of a (possibly multimode) phase-space function
    nr = len(w1)
    dim = len(w1.shape)
    dx = (rl / (nr - 1))
    vol = dx**dim
    p, q = relative_lorenz_increasing(w1, w2)
    return p * vol, q * vol
