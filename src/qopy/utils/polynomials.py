import numpy as np
import scipy
import math


def hermite(n, x):
    p = np.append(np.zeros(n), 1)
    f = np.polynomial.hermite.hermval(x, p)
    return f


def laguerre(n, x, alpha=0):
    fx = scipy.special.genlaguerre(n, alpha)(x)
    return fx


def hermite_bivariate(n, x, y):
    # Note that hermite(n, x) = hermite_bivariate(n, 2*x, -1)
    # See https://doi.org/10.1016/j.aml.2012.02.043
    res = 0
    for k in range(int(n/2)+1):
        res += x**(n-2*k)*y**k/(scipy.special.factorial(n-2*k)*scipy.special.factorial(k))
    return scipy.special.factorial(n)*res


def hermite_2index(m, n, x, y, w, z, tau):
    res = 0
    for k in range(min(m, n)+1):
        res += (scipy.special.factorial(m)*scipy.special.factorial(n)/(scipy.special.factorial(m-k)*scipy.special.factorial(n-k)*scipy.special.factorial(k))) * tau ** k * hermite_bivariate(m - k, x, y) * hermite_bivariate(n - k, w, z)
    return res
