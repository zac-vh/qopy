import numpy as np
import scipy

def phase_distribution_tensor(w1, w2):
    # use the convention to use order |0,1><2,3|=|0><2|o|1><3|
    # to trace over second mode, trace over indices 1,3
    # to trace over first mode, trace over indices 0,2
    return np.tensordot(w1, w2, axes=0)


def phase_distribution_integrate_2mode(W, rl):
    # Integrate the 2-mode Wigner function
    nr = len(W)
    x = np.linspace(-rl / 2, rl / 2, nr)
    return scipy.integrate.simpson(scipy.integrate.simpson(scipy.integrate.simpson(scipy.integrate.simpson(W, x), x=x), x=x), x=x)


def phase_distribution_partial_trace(W, rl, mode=2):
    nr = len(W)
    x = np.linspace(-rl/2, rl/2, nr)
    if mode==2:
        w = scipy.integrate.simpson(scipy.integrate.simpson(W, x=x), x=x)
    if mode==1:
        w = scipy.integrate.simpson(scipy.integrate.simpson(W, x=x, axis=0), x=x, axis=0)
    return w