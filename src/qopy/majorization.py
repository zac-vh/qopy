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


def vec_major(v1, v2, tol=0):
    len1 = len(v1)
    len2 = len(v2)
    if len1 > len2:
        v2 = np.append(v2, np.zeros(len1 - len2))
    elif len2 > len1:
        v1 = np.append(v1, np.zeros(len2 - len1))
    v1d = np.sort(v1)[::-1]
    v2d = np.sort(v2)[::-1]
    v1dc = np.cumsum(v1d)
    v2dc = np.cumsum(v2d)
    diff = np.min(v1dc - v2dc)
    maj = False
    if diff >= -tol:
        maj = True
    return maj


def vec_dec(v, n=0):
    lenv = len(v)
    if n < 0:
        return
    if n == 0:
        return np.sort(v)[::-1]
    if n >= (lenv - 1):
        return v
    v_hart = v[:n]
    v_end = v[n:]
    v_end_dec = np.sort(v_end)[::-1]
    v_dec = np.append(v_hart, v_end_dec)
    return v_dec

