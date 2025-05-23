import numpy as np
import scipy
import math

def poly_hermite(n, x):
    fx = 2 ** n * np.sqrt(math.pi) * (scipy.special.hyp1f1(-n / 2, 1 / 2, x ** 2) / scipy.special.gamma(
        (1 - n) / 2) - 2 * x * scipy.special.hyp1f1((1 - n) / 2, 3 / 2, x ** 2) / scipy.special.gamma(-n / 2))
    return fx

def poly_laguerre(n, x, alpha=0):
    fx =  scipy.special.genlaguerre(n, alpha)(x)
    return fx