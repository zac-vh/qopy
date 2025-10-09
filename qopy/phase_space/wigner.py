import numpy as np
import qopy.phase_space.cross_wigner as xwig
from qopy.utils.grid import square
import mpmath
import scipy

def fock(n, rl, nr):
    # Wigner function of the nth Fock states
    # The Wigner function is a matrix (nr x nr) defined over [-rl/2, rl/2]x[-rl/2, rl/2]
    return np.real(xwig.fock(n, n, rl, nr))


def gaussian(alpha, xi, rl, nr):
    return np.real(xwig.gaussian(alpha, alpha, xi, xi, rl, nr))


def gaussian_fock(alpha, xi, n, rl, nr):
    return np.real(xwig.gaussian_fock(alpha, alpha, xi, xi, n, n, rl, nr))


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


def gkp(s, rl, nr):
    # Compute the Wigner function of the finite-energy GKP
    # sigma should be taken in the range 0.3-1
    mx, mp = square(rl, nr)
    jtheta = np.vectorize(mpmath.jtheta, 'D')
    C = jtheta(3, 0, np.exp(-4 * np.pi * s ** 2)) * jtheta(3, 0, np.exp(-4 * np.pi * ((s ** 4 + 1) / s ** 2))) \
        + jtheta(2, 0, np.exp(-4 * np.pi * s ** 2)) * jtheta(2, 0, np.exp(-4 * np.pi * ((s ** 4 + 1) / s ** 2)))

    w0 = (1 / (4 * np.pi * np.sqrt(s ** 4 + 1))) * np.exp(-mx ** 2 * (s ** 2 / (s ** 4 + 1))) * np.exp(
        -mp ** 2 * ((s ** 4 + 1) / s ** 2))
    w1 = jtheta(3, (-np.sqrt(np.pi) / 2) * mx / (s ** 4 + 1),
                np.exp(-(np.pi / 4) * (s ** 2 / (s ** 4 + 1)))) * jtheta(3,
                                                                           -1j * (np.sqrt(np.pi) / 2) * mp / s ** 2,
                                                                           np.exp(-np.pi / (4 * s ** 2)))
    w2 = jtheta(3, (-np.sqrt(np.pi) / 2) * mx / (s ** 4 + 1) + np.pi / 2,
                np.exp(-(np.pi / 4) * (s ** 2 / (s ** 4 + 1)))) * jtheta(3, -1j * (
                np.sqrt(np.pi) / 2) * mp / s ** 2 + np.pi / 2, np.exp(-np.pi / (4 * s ** 2)))

    w = C ** (-1) * w0 * (w1 + w2)
    return w


def particle_in_a_box(a, rl, nr):
    mx, mp = square(rl, nr)
    mxa = mx[np.abs(mx) < a/2]
    mpa = mp[np.abs(mx) < a/2]
    w = np.zeros([nr, nr])
    w[np.abs(mx) < a/2] = (1/np.pi)*(1-2*(np.abs(mxa)/a))*np.sinc(mpa*(a-2*np.abs(mxa))/np.pi)
    return w


def cubic_phase(gamma, rl, nr, disp=(0, 0), sq=1):
    # Compute the Wigner function of a cubic-phase-gate acting on vacuum
    # The phase is exp(i*gamma*x^3)
    # See https://doi.org/10.1103/PhysRevA.100.013831
    if gamma == 0:
        return fock(0, rl, nr)
    mx, mp = square(rl, nr)
    mx = (mx-disp[0])/sq
    mp = (mp-disp[1])*sq
    mgamma = np.exp(1/(54*gamma**2))*(4/(3*np.abs(gamma)))**(1/3)/np.sqrt(np.pi)
    airy = scipy.special.airy((4/(3*gamma))**(1/3)*(3*gamma*mx**2-mp+1/(12*gamma)))[0]
    wcubic = mgamma*np.exp(-mp/(3*gamma))*airy
    return wcubic