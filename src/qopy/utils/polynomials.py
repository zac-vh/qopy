import numpy as np
import scipy
import math

def poly_hermite(n, x):
    p = np.append(np.zeros(n), 1)
    f = np.polynomial.hermite.hermval(x, p)
    return f

def poly_laguerre(n, x, alpha=0):
    fx =  scipy.special.genlaguerre(n, alpha)(x)
    return fx