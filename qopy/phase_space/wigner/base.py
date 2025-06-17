import numpy as np
import scipy
import qopy.phase_space.cross_wigner as xwig


def fock_radial(n, r):
    # Radial function of the Wigner function of the nth Fock state
    p = np.append(np.zeros(n), 1)
    wr = (-1) ** n * (1 / np.pi) * np.multiply(np.exp(-r ** 2), np.polynomial.laguerre.lagval(2 * r ** 2, p))
    return wr


def fock_area(n, a):
    # Area function associated to the Wigner function of the nth Fock state
    # F(a) = pi*W(sqrt(a))
    p = np.append(np.zeros(n), 1)
    wr = (-1) ** n *np.multiply(np.exp(-a), np.polynomial.laguerre.lagval(2 * a, p))
    return wr


def fock(n, rl, nr):
    # Wigner function of the nth Fock states
    # The Wigner function is a matrix (nr x nr) defined over [-rl/2, rl/2]x[-rl/2, rl/2]
    return xwig.fock.grid(n, n, rl, nr)


def fock_fn(n):
    lag = scipy.special.laguerre(n)
    def W(x, p):
        r2 = x ** 2 + p ** 2
        return (-1) ** n * (1 / np.pi) * np.exp(-r2) * lag(2 * r2)

def gaussian(alpha, xi, rl, nr):
    return np.real(xwig.gaussian.grid(alpha, alpha, xi, xi, rl, nr))


def gaussian_fock(alpha, xi, n, rl, nr):
    return np.real(xwig.gaussian_fock.grid(alpha, alpha, xi, xi, n, n, rl, nr))


def gaussian_via_covariance_matrix(rl, nr, alpha=0, covmat=np.eye(2) / 2):
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


def from_psi(psi, rl, nr):
    # Wigner function of the wave-function psi
    # psi should be sampled on xl (see get_xl)
    return np.real(xwig.from_psi(psi, psi, rl, nr))
