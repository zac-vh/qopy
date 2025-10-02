import scipy
import numpy as np
import math

def integrate_1d(func, xl):
    return scipy.integrate.simpson(func, x=xl)


def shannon_entropy_1d(rx, rl):
    nr = len(rx)
    x = np.linspace(-rl / 2, rl / 2, nr)
    rxlog = np.zeros(nr)
    rxlog[np.nonzero(rx)] = np.log(np.abs(rx[np.nonzero(rx)]))
    return - scipy.integrate.simpson(rx * rxlog, x=x)


def renyi_entropy_1d(rx, rl, alpha):
    nr = len(rx)
    x = np.linspace(-rl / 2, rl / 2, nr)
    if alpha == 1:
        return shannon_entropy_1d(rx, rl)
    if math.isinf(alpha):
        m = np.max(np.abs(rx))
        return -np.log(m)
    if alpha == 0:
        rxa = (np.abs(rx) > 0)
        return np.log(scipy.integrate.simpson(rxa, x=x))
    rxa = np.abs(rx) ** alpha
    return (1 / (1 - alpha)) * np.log(scipy.integrate.simpson(rxa, x=x))


def fourier_transform_1d(fx, xl):
    nrl = len(xl)
    phi = np.zeros(nrl, dtype=complex)
    for i in range(nrl):
        pi = xl[i]
        re = scipy.integrate.simpson(np.real(fx*np.exp(-1j*pi*xl)), x=xl)/np.sqrt(2*math.pi)
        im = scipy.integrate.simpson(np.imag(fx*np.exp(-1j*pi*xl)), x=xl)/np.sqrt(2*math.pi)
        phi[i] = re + im * 1j
    return phi