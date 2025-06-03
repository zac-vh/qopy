import numpy as np
from qopy.utils.grid import grid_square
import mpmath
import scipy
import qopy.phase_space.cross_wigner as xwig
from qopy.phase_space.wavefunction import get_xl, ket_to_psi
from qopy.utils.linalg import matrix_eigenvalues_eigenvectors


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


def gkp(s, rl, nr):
    # Compute the Wigner function of the finite-energy GKP
    # sigma should be taken in the range 0.3-1
    mx, mp = grid_square(rl, nr)
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
    mx, mp = grid_square(rl, nr)
    mxa = mx[np.abs(mx) < a/2]
    mpa = mp[np.abs(mx) < a/2]
    w = np.zeros([nr, nr])
    w[np.abs(mx) < a/2] = (1/np.pi)*(1-2*(np.abs(mxa)/a))*np.sinc(mpa*(a-2*np.abs(mxa))/np.pi)
    return w


def cubic_phase_state(gamma, rl, nr, disp=(0, 0), sq=1):
    # Compute the Wigner function of a cubic-phase-gate acting on vacuum
    if gamma == 0:
        return fock(0, rl, nr)
    mx, mp = grid_square(rl, nr)
    mx = (mx-disp[0])/sq
    mp = (mp-disp[1])*sq
    mgamma = np.exp(1/(54*gamma**2))*(4/(3*np.abs(gamma)))**(1/3)/np.sqrt(math.pi)
    airy = scipy.special.airy((4/(3*gamma))**(1/3)*(3*gamma*mx**2-mp+1/(12*gamma)))[0]
    wcubic = mgamma*np.exp(-mp/(3*gamma))*airy
    return wcubic


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


def from_density(rho, rl, nr, isherm=True):
    n = rho.shape[0]
    w = np.zeros((nr, nr), dtype=complex)
    for i in range(n):
        for j in range(i+1 if isherm else n):
            rij = rho[i, j]
            if rij == 0:
                continue
            wij = xwig.fock.grid(i, j, rl, nr)
            if isherm:
                if i != j:
                    w += 2*np.real(rij * wij)
                if i == j:
                    w += rij * wij
            else:
                w += rij * wij
    return np.real(w) if isherm else w


def from_density_via_set(rho, wijset, isherm=True):
    # Build the Wigner function of the density matrix rho (in wij_set basis)
    n = rho.shape[0]
    w = np.zeros(wijset.shape[2:], dtype=complex)
    if isherm:
        for i in range(n):
            w += rho[i, i] * wijset[i, i]
            for j in range(i+1, n):
                rij = rho[i, j]
                if rij == 0:
                    continue
                w += 2*np.real(rij * wijset[i, j])
        w = np.real(w)
    else:
        for i in range(n):
            for j in range(n):
                rij = rho[i, j]
                if rij == 0:
                    continue
                w += rij * wijset[i, j]
    return w


def from_density_via_psi(rho, rl, nr):
    # Compute the Wigner function from the wavefunction of the eigenvectors of rho
    # More efficient when N >> 1
    xl = get_xl(rl, nr)
    w = np.zeros((nr, nr), dtype=complex)
    eigvals, eigkets = matrix_eigenvalues_eigenvectors(rho)
    N = len(eigvals)
    for j in range(N):
        psij = ket_to_psi(eigkets[j], xl)
        w += eigvals[j] * from_psi(psij, rl, nr)
    return np.real(w)