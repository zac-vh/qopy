'''
Contains all the functions extracting information from (density) operators

'''

import numpy as np


def mean(rho):
    n = len(rho)
    a = 0+0j
    for i in range(n-1):
        a = a + rho[i, i+1]*np.sqrt(i+1)
    xm = np.sqrt(2)*np.real(a)
    pm = -np.sqrt(2)*np.imag(a)
    return np.array([xm, pm])


def covariance_matrix(rho, mean=None):
    n = len(rho)
    if mean is None:
        [xm, pm] = mean(rho)
    elif not (isinstance(mean, list) or (isinstance(mean, np.ndarray)) or (isinstance(mean, tuple))):
        xm = np.sqrt(2)*np.real(mean)
        pm = np.sqrt(2)*np.imag(mean)
    else:
        xm = mean[0]
        pm = mean[1]

    nm = photon_number(rho)
    aa = 0+0j
    for i in range(n-2):
        aa = aa + rho[i, i+2]*np.sqrt((i+1)*(i+2))
    xxm = np.real(aa)+nm+1/2-xm**2
    ppm = -np.real(aa)+nm+1/2-pm**2
    xpm = -np.imag(aa)-xm*pm
    cov = np.array([[xxm, xpm], [xpm, ppm]])
    return cov


def von_neumann_entropy(rho):
    la = np.real(np.linalg.eig(rho)[0])
    la = np.delete(la, np.where(la <= 0))
    return -np.sum(la * np.log(la))


def renyi_entropy(rho, a=1):
    # Compute the Von Neumann entropy (or Rényi entropy) of rho
    # Eigenvalues of rho are assumed positive
    la = np.real(np.linalg.eig(rho)[0])
    la = np.delete(la, np.where(la <= 0))
    if a == 0:
        return np.log(np.sum(np.abs(la) != 0))
    if a == 1:
        return von_neumann_entropy(rho)
    if np.isinf(a):
        return - np.log(np.max(np.abs(la)))
    return (1 / (1 - a)) * np.log(np.sum(la ** a))


def photon_number(rho):
    d = np.diag(range(len(rho)))
    return np.real(np.trace(d @ rho))


def energy(rho):
    d = np.diag(range(len(rho)))
    return np.real(np.trace(d@rho))+1/2


def purity(rho):
    return np.trace(rho @ rho)

