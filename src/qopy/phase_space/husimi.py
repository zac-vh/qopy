import numpy as np
import math


def fock_cross_husimi(i, j, rl, nr, hbar=1):
    #choose hbar=1/2 to have q0=w0
    x = np.linspace(-rl/2, rl/2, nr)
    mx, mp = np.meshgrid(x, x, indexing='ij')
    ma = (mx+1j*mp)/np.sqrt(2*hbar)
    fact = 2*hbar*math.pi
    hij = np.exp(-np.abs(ma)**2)*np.conj(ma)**i*ma**j/np.sqrt(float(math.factorial(i)*math.factorial(j)))
    return hij/fact


def fock_husimi(n, rl, nr, hbar=1):
    return np.real(fock_cross_husimi(n, n, rl, nr, hbar))


def density_to_husimi(rho, rl, nr, isherm=True, hbar=1):
    # Build the Husimi function of the density matrix rho
    n = len(rho)
    h = np.zeros([nr, nr])
    if isherm:
        for i in range(n):
            rii = rho[i][i]
            if rii != 0:
                h = h + rii * fock_cross_husimi(i, i, rl, nr, hbar)
            for j in range(i + 1, n):
                rij = rho[i][j]
                if rij != 0:
                    rho_wij = rho[i][j] * fock_cross_husimi(i, j, rl, nr, hbar)
                    h = h + rho_wij + np.conj(rho_wij)
        h = np.real(h)
    else:
        for i in range(n):
            for j in range(n):
                rij = rho[i][j]
                if rij != 0:
                    h = h + rho[i][j] * fock_cross_husimi(i, j, rl, nr, hbar)
    return h
