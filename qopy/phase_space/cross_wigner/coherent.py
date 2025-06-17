import numpy as np
from qopy.utils.grid import grid_square

def fn_xp(alpha, beta, x, p):
    re = (alpha+np.conj(beta))/2
    im = (alpha-np.conj(beta))/(2*1j)
    exp = -(1/2)*np.abs(alpha)**2-(1/2)*np.abs(beta)**2+alpha*np.conj(beta)\
          -(x-np.sqrt(2)*re)**2-(p-np.sqrt(2)*im)**2
    return (1/np.pi)*np.exp(exp)


def fn(alpha, beta):
    """
    Returns a fast callable Wigner function of the cross-Wigner between
    D(alpha)|0> and D(beta)|0>, assuming x and p are numpy arrays (e.g. meshgrids).
    """
    re = (alpha + np.conj(beta)) / 2
    im = (alpha - np.conj(beta)) / (2j)
    prefactor = (1 / np.pi) * np.exp(
        -0.5 * (np.abs(alpha) ** 2 + np.abs(beta) ** 2) + alpha * np.conj(beta)
    )
    dx = np.sqrt(2) * re
    dp = np.sqrt(2) * im
    def W(x, p):
        # expects x and p as ndarrays, typically from np.meshgrid
        xdiff2 = (x - dx) ** 2
        pdiff2 = (p - dp) ** 2
        return prefactor * np.exp(-(xdiff2 + pdiff2))
    return W


def grid(alpha, beta, rl, nr):
    mx, mp = grid_square(rl, nr)
    wij = fn_xp(alpha, beta, mx, mp)
    return wij