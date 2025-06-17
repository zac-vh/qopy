import numpy as np
import scipy
from qopy.utils.grid import grid_square


def fn_xp(i, j, x, p):
    # Return the Wigner function of |i><j| evaluated at (x,p)
    # x, p should have the same dimension
    r = np.sqrt(x ** 2 + p ** 2)
    if j == i:
        n = i
        w = (-1) ** n * (1 / np.pi) * np.exp(-r ** 2)*scipy.special.laguerre(n)(2 * r ** 2)
        return w
    if i < j:
        m, n = i, j
        w = (1 / np.pi) * np.exp(-r ** 2) * (-1) ** m * np.sqrt(2) ** (n - m) \
            * np.sqrt(scipy.special.factorial(m) / scipy.special.factorial(n)) * \
            (x + 1j * p) ** (n - m) * scipy.special.genlaguerre(m, n - m)(2 * r ** 2)
        return w
    if i > j:
        m, n = j, i
        w = (1 / scipy.special.pi) * np.exp(-r ** 2) * (-1) ** m * np.sqrt(2) ** (n - m) \
            * np.sqrt(scipy.special.factorial(m) / scipy.special.factorial(n)) * \
            (x - 1j * p) ** (n - m) * scipy.special.genlaguerre(m, n - m)(2 * r ** 2)
        return w


def fn(i, j):
    """
    Retourne une fonction W(x, p) correspondant à la Wigner function de |i><j|.
    x et p peuvent être scalaires ou tableaux NumPy de même forme.
    """
    if i == j:
        n = i
        lag = scipy.special.laguerre(n)

        def W(x, p):
            r2 = x**2 + p**2
            return (-1)**n * (1 / np.pi) * np.exp(-r2) * lag(2 * r2)
    elif i < j:
        m, n = i, j
        genlag = scipy.special.genlaguerre(m, n - m)
        coef = (1 / np.pi) * (-1)**m * (np.sqrt(2)**(n - m)) \
               * np.sqrt(scipy.special.factorial(m) / scipy.special.factorial(n))

        def w(x, p):
            r2 = x**2 + p**2
            return coef * np.exp(-r2) * (x + 1j * p)**(n - m) * genlag(2 * r2)
    else:  # i > j
        m, n = j, i
        genlag = scipy.special.genlaguerre(m, n - m)
        coef = (1 / np.pi) * (-1)**m * (np.sqrt(2)**(n - m)) \
               * np.sqrt(scipy.special.factorial(m) / scipy.special.factorial(n))

        def w(x, p):
            r2 = x**2 + p**2
            return coef * np.exp(-r2) * (x - 1j * p)**(n - m) * genlag(2 * r2)

    return w


def grid(i, j, rl, nr):
    mx, mp = grid_square(rl, nr)
    wij = fn_xp(i, j, mx, mp)
    return wij


def grid_set(N, rl, nr):
    wijset = np.empty([N, N, nr, nr], dtype=complex)
    for i in range(N):
        wijset[i, i] = grid(i, i, rl, nr)
        for j in range(i + 1, N):
            wij = grid(i, j, rl, nr)
            wijset[i, j] = wij
            wijset[j, i] = np.conj(wij)
    return wijset


