import numpy as np
import scipy
import qopy.phase_space.cross_wigner as xwig

def radial(n, r):
    # Radial function of the Wigner function of the nth Fock state
    p = np.append(np.zeros(n), 1)
    wr = (-1) ** n * (1 / np.pi) * np.multiply(np.exp(-r ** 2), np.polynomial.laguerre.lagval(2 * r ** 2, p))
    return wr


def area(n, a):
    # Area function associated to the Wigner function of the nth Fock state
    # F(a) = pi*W(sqrt(a))
    p = np.append(np.zeros(n), 1)
    wr = (-1) ** n *np.multiply(np.exp(-a), np.polynomial.laguerre.lagval(2 * a, p))
    return wr


def grid(n, rl, nr):
    # Wigner function of the nth Fock states
    # The Wigner function is a matrix (nr x nr) defined over [-rl/2, rl/2]x[-rl/2, rl/2]
    return xwig.fock.grid(n, n, rl, nr)


def fn(n):
    lag = scipy.special.laguerre(n)
    def w(x, p):
        r2 = x ** 2 + p ** 2
        return (-1) ** n * (1 / np.pi) * np.exp(-r2) * lag(2 * r2)
    return w
