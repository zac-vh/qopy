import numpy as np
import scipy

def wig2m_int(W, rl):
    # Integrate the 2-mode Wigner function
    nr = len(W)
    x = np.linspace(-rl / 2, rl / 2, nr)
    return scipy.integrate.simpson(scipy.integrate.simpson(scipy.integrate.simpson(scipy.integrate.simpson(W, x), x=x), x=x), x=x)


def wig2m_ptrace(W, rl, mode=2):
    nr = len(W)
    x = np.linspace(-rl/2, rl/2, nr)
    if mode==2:
        w = scipy.integrate.simpson(scipy.integrate.simpson(W, x=x), x=x)
    if mode==1:
        w = scipy.integrate.simpson(scipy.integrate.simpson(W, x=x, axis=0), x=x, axis=0)
    return w


def wig2m_h(W, rl):
    # Compute the Wigner entropy of a 2-mode Wigner function
    nr = len(W)
    x = np.linspace(-rl / 2, rl / 2, nr)
    Wlog = np.zeros([nr, nr, nr, nr])
    Wlog[np.nonzero(W)] = np.log(np.abs(W[np.nonzero(W)]))
    return - wig2m_int(W * Wlog, rl)