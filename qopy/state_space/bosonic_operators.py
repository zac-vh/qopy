import numpy as np
from scipy.linalg import expm


def annihilation(N):
    return np.diag(np.sqrt(np.arange(1, N)), k=1).astype(complex)


def creation(N):
    return np.diag(np.sqrt(np.arange(1, N)), k=-1).astype(complex)


def position(N):
    return (annihilation(N)+creation(N))/np.sqrt(2)


def momentum(N):
    return 1j*(creation(N)-annihilation(N))/np.sqrt(2)


def quadrature(N, theta):
    return np.cos(theta)*position(N)+np.sin(theta)*momentum(N)


def rotation(N, theta):
    return np.diag(np.exp(-1j * theta * np.arange(N)))


def squeezing(N, r):
    a = annihilation(N)
    adag = annihilation(N)
    G = (1/2) * (np.conj(r) * (a @ a) - r * (adag @ adag))
    return expm(G)


def displacement(N, alpha):
    G = alpha*creation(N) - np.conj(alpha)*annihilation(N)
    return expm(G)


def parity(N, alpha=0):
    par = np.diag((-1)**np.arange(N)).astype(complex)
    if alpha != 0:
        par = displacement(N, alpha) @ par @ displacement(N, -alpha)
    return np.diag((-1)**np.arange(N)).astype(complex)


def photon_number(N):
    return np.diag(np.arange(N)).astype(complex)


def hamiltonian(N):
    return np.diag(np.arange(N)+1/2).astype(complex)