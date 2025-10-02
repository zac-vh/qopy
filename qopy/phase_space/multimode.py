import numpy as np
import scipy

def wigner_tensor(w1, w2):
    # use the convention to use order W(x_1,p_1,x_2,p_2)=W(x_1,p_1)W(x_2,p_2)
    return np.tensordot(w1, w2, axes=0)


def wigner_integrate_2mode(W, rl):
    # Integrate the 2-mode Wigner function
    nr = len(W)
    x = np.linspace(-rl / 2, rl / 2, nr)
    return scipy.integrate.simpson(scipy.integrate.simpson(scipy.integrate.simpson(scipy.integrate.simpson(W, x), x=x), x=x), x=x)


def wigner_partial_trace(W, rl, mode=2):
    nr = len(W)
    x = np.linspace(-rl/2, rl/2, nr)
    if mode==2:
        w = scipy.integrate.simpson(scipy.integrate.simpson(W, x=x), x=x)
    if mode==1:
        w = scipy.integrate.simpson(scipy.integrate.simpson(W, x=x, axis=0), x=x, axis=0)
    return w