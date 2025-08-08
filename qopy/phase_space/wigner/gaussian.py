import numpy as np
import qopy.phase_space.cross_wigner as xwig


def grid(alpha, xi, rl, nr):
    return np.real(xwig.gaussian.grid(alpha, alpha, xi, xi, rl, nr))


def grid_fock(alpha, xi, n, rl, nr):
    return np.real(xwig.gaussian_fock.grid(alpha, alpha, xi, xi, n, n, rl, nr))


def grid_from_covariance_matrix(rl, nr, alpha=0, covmat=np.eye(2) / 2):
    # Wigner function of a Gaussian state with mean d and covariance matrix gamma
    if not (isinstance(alpha, list) or (isinstance(alpha, np.ndarray)) or (isinstance(alpha, tuple))):
        d = np.sqrt(2)*np.array([np.real(alpha), np.imag(alpha)])
    else:
        d = alpha
    [g11, g12], [g21, g22] = np.linalg.inv(covmat)
    dx, dy = d
    x = np.linspace(-rl / 2, rl / 2, nr)
    [mx, my] = np.meshgrid(x, x, indexing='ij')
    a = g11 * (mx - dx) ** 2 + (g21 + g12) * (mx - dx) * (my - dy) + g22 * (my - dy) ** 2
    w = np.exp(-a / 2) / (2 * np.pi * np.sqrt(np.linalg.det(covmat)))
    return w