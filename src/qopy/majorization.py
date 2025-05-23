import numpy as np




def wig_cum(w, rl=None, arrow='dec'):
    # Compute the discrete cumulative sum of the Wigner function
    nr = len(w)
    if rl is None:
        rl = nr
    v = np.reshape(w, nr ** 2)
    if arrow == 'dec':
        vd = np.sort(v)[::-1]
    if arrow == 'inc':
        vd = np.sort(v)
    if arrow == 'pos':
        vd = np.sort(np.maximum(np.zeros(nr ** 2), v))[::-1]
    if arrow == 'neg':
        vd = np.sort(np.minimum(np.zeros(nr ** 2), v))
    vdc = np.cumsum(vd) * (rl / (nr-1)) ** 2
    return vdc


def wig_major(w1, w2, rl=None, tol=0):
    # Check whether w1 majorizes w2
    # tol is a tolerance value
    w1cum = wig_cum(w1, rl)
    w2cum = wig_cum(w2, rl)
    diff = np.min(w1cum - w2cum)
    maj = False
    if diff >= -tol:
        maj = True
    return maj


def major(p, q, tol=0):
    # Check whether p majorizes q
    # tol is a tolerance value
    lp = len(p)
    lq = len(q)
    l = np.max([lp, lq])
    p = np.concatenate([p, np.zeros(l-lp)])
    q = np.concatenate([q, np.zeros(l-lq)])
    pcum = np.cumsum(np.sort(p)[::-1])
    qcum = np.cumsum(np.sort(q)[::-1])
    diff = np.min(pcum - qcum)
    maj = False
    if diff >= -tol:
        maj = True
    return maj


def wig_rear(w, rear='dec'):
    nr = len(w)
    wr = np.zeros([nr, nr])
    wd = np.sort(np.reshape(w, nr**2))
    if rear == 'dec':
        wd = wd[::-1]
    elif rear == 'pos':
            wd = np.maximum(np.zeros(nr**2), wd)
            wd = wd[::-1]
    elif rear == 'neg':
            wd = np.minimum(np.zeros(nr**2), wd)
    x = np.linspace(-1, 1, nr)
    mx, my = np.meshgrid(x, x)
    m = mx ** 2 + my ** 2
    ind = np.unravel_index(np.argsort(m, axis=None), m.shape)
    wr[ind] = wd
    return wr
