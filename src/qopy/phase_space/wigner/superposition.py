import numpy as np
import qopy.phase_space.cross_wigner as xwig
from qopy import overlap

def coherent(alphas, cis, rl, nr, normalized=True):
    w = np.zeros([nr, nr])
    n = len(alphas)
    norm = 0
    for i in range(n):
        ai = alphas[i]
        ci = cis[i]
        w = w + np.abs(ci) ** 2 * xwig.coherent.grid(ai, ai, rl, nr)
        norm = norm + np.abs(ci)**2
        for j in range(i+1, n):
            aj = alphas[j]
            cj = cis[j]
            w = w + 2*np.real(ci * np.conj(cj) * xwig.coherent.grid(ai, aj, rl, nr))
            norm = norm + 2*np.real(np.conj(cj)*ci*overlap.coherent(ai, aj))
    w = np.real(w)
    if normalized:
        w = w/norm
    return w


def gaussian(alpha_list, xi_list, amplitude_list, rl, nr, normalized=True):
    w = np.zeros([nr, nr])
    n = len(alpha_list)
    norm = 0
    for i in range(n):
        ai = alpha_list[i]
        xi = xi_list[i]
        ci = amplitude_list[i]
        w = w + np.abs(ci) ** 2 * xwig.gaussian.grid(ai, ai, xi, xi, rl, nr)
        norm = norm + np.abs(ci)**2
        for j in range(i+1, n):
            aj = alpha_list[j]
            xj = xi_list[j]
            cj = amplitude_list[j]
            w = w + 2*np.real(ci * np.conj(cj) * xwig.gaussian.grid(ai, aj, xi, xj, rl, nr))
            norm = norm + 2*np.real(np.conj(cj)*ci*overlap.gaussian(ai, aj, xi, xj))
    w = np.real(w)
    if normalized:
        w = w/norm
    return w


def gaussian_fock(alpha_list, xi_list, n_list, amplitude_list, rl, nr, normalized=True):
    w = np.zeros([nr, nr])
    n = len(amplitude_list)
    norm = 0
    for i in range(n):
        ai = alpha_list[i]
        xi = xi_list[i]
        ni = n_list[i]
        ci = amplitude_list[i]
        w = w + np.abs(ci) ** 2 * xwig.gaussian_fock.grid(ai, ai, xi, xi, ni, ni, rl, nr)
        norm = norm + np.abs(ci)**2
        for j in range(i+1, n):
            aj = alpha_list[j]
            xj = xi_list[j]
            nj = n_list[j]
            cj = amplitude_list[j]
            w = w + 2*np.real(ci * np.conj(cj) * xwig.gaussian_fock.grid(ai, aj, xi, xj, ni, nj, rl, nr))
            norm = norm + 2*np.real(np.conj(cj)*ci*overlap.gaussian_fock(ai, aj, xi, xj, ni, nj))
    w = np.real(w)
    if normalized:
        w = w/norm
    return w