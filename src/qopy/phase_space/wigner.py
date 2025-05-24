"""
qopy.wigner
============

Tools for creating Wigner functions

F(real) -> W

"""


import numpy as np
import math
import scipy
import mpmath
from qopy.utils.grid import grid_square as grid

def wigner_radial_fock(n, r):
    # Radial function of the Wigner function of the nth Fock state
    p = np.append(np.zeros(n), 1)
    wr = (-1) ** n * (1 / math.pi) * np.multiply(np.exp(-r ** 2), np.polynomial.laguerre.lagval(2 * r ** 2, p))
    return wr


def wigner_area_fock(n, a):
    # Area function associated to the Wigner function of the nth Fock state
    p = np.append(np.zeros(n), 1)
    wr = (-1) ** n *np.multiply(np.exp(-a), np.polynomial.laguerre.lagval(2 * a, p))
    return wr


def wigner_fock(n, rl, nr):
    # Wigner function of the nth Fock states
    # The Wigner function is a matrix (nr x nr) defined over [-rl/2, rl/2]x[-rl/2, rl/2]
    return cross_wigner_fock(n, n, rl, nr)


def cross_wigner_fock_xp(i, j, x, p):
    # Return the Wigner function of |i><j| evaluated at (x,p)
    # x, p should have the same dimension
    r = np.sqrt(x ** 2 + p ** 2)
    if j == i:
        n = i
        w = (-1) ** n * (1 / math.pi) * np.exp(-r ** 2)*scipy.special.laguerre(n)(2 * r ** 2)
        return w
    if i < j:
        m, n = i, j
        w = (1 / math.pi) * np.exp(-r ** 2) * (-1) ** m * np.sqrt(2) ** (n - m) \
            * np.sqrt(math.factorial(m) / math.factorial(n)) * \
            (x + 1j * p) ** (n - m) * scipy.special.genlaguerre(m, n - m)(2 * r ** 2)
        return w
    if i > j:
        m, n = j, i
        w = (1 / math.pi) * np.exp(-r ** 2) * (-1) ** m * np.sqrt(2) ** (n - m) \
            * np.sqrt(math.factorial(m) / math.factorial(n)) * \
            (x - 1j * p) ** (n - m) * scipy.special.genlaguerre(m, n - m)(2 * r ** 2)
        return w


def cross_wigner_fock(i, j, rl, nr):
    mx, mp = grid(rl, nr)
    wij = cross_wigner_fock_xp(i, j, mx, mp)
    return wij


def cross_wigner_fock_set(N, rl, nr):
    wijset = np.zeros([N, N, nr, nr], dtype=complex)
    for i in range(N):
        wijset[i][i] = wigner_fock(i, rl, nr)
        for j in range(i + 1, N):
            wij = cross_wigner_fock(i, j, rl, nr)
            wijset[i][j] = wij
            wijset[j][i] = np.conj(wij)
    return wijset


def cross_wigner_fock_displaced_xp(m, n, alpha, beta, x, p):
    # Return the Wigner function of the operator D(alpha)|m><n|D(-beta)
    gamma = np.sqrt(2)*(x+1j*p)
    delta = gamma-alpha-beta
    sum = 0
    for p in range(min(m, n)+1):
        sum = sum + (-1)**p/(math.factorial(p)*math.factorial(n-p)*math.factorial(m-p))*delta**(n-p)*np.conj(delta)**(m-p)

    exp = np.exp(-(1/2)*(np.abs(gamma)**2+np.abs(alpha)**2+np.abs(beta)**2))*np.exp(-alpha*np.conj(beta)+gamma*np.conj(beta)+alpha*np.conj(gamma))
    return (np.sqrt(math.factorial(m)*math.factorial(n))/math.pi)*exp*sum


def cross_wigner_fock_displaced(m, n, alpha, beta, rl, nr):
    # Return the Wigner function of the operator D(alpha)|m><n|D(-beta)
    mx, mp = grid(rl, nr)
    wij = cross_wigner_fock_displaced_xp(m, n, alpha, beta, mx, mp)
    return wij


def cross_wigner_coherent_xp(alpha, beta, x, p):
    re = (alpha+np.conj(beta))/2
    im = (alpha-np.conj(beta))/(2*1j)
    exp = -(1/2)*np.abs(alpha)**2-(1/2)*np.abs(beta)**2+alpha*np.conj(beta)\
          -(x-np.sqrt(2)*re)**2-(p-np.sqrt(2)*im)**2
    return (1/math.pi)*np.exp(exp)


def cross_wigner_coherent(a, b, alpha):
    # return the complex Wigner function of |a><b|
    return (2/math.pi)*np.exp(-2*np.abs(alpha-(a+b)/2)**2)*np.exp(1j*np.imag(2*alpha*np.conj(b-a)))*np.exp(1j*np.imag(b*np.conj(a)))


def cross_wigner_coherent_squeezed(alpha, beta, rl, nr, sq=1):
    mx, mp = grid(rl, nr)
    wij = cross_wigner_coherent_xp(alpha, beta, sq * mx, mp / sq)
    return wij


def wigner_coherent_superposition_squeezed(alphas, cis, rl, nr, sq=1, normalized=True):
    w = np.zeros([nr, nr])
    n = len(alphas)
    norm = 0
    for i in range(n):
        ai = alphas[i]
        ci = cis[i]
        w = w + np.abs(ci) ** 2 * cross_wigner_coherent_squeezed(ai, ai, rl, nr, sq)
        norm = norm + np.abs(ci)**2
        for j in range(i+1, n):
            aj = alphas[j]
            cj = cis[j]
            w = w + 2*np.real(ci * np.conj(cj) * cross_wigner_coherent_squeezed(ai, aj, rl, nr, sq))
            norm = norm + 2*np.real(np.conj(cj)*ci*np.exp(-(1/2)*(np.abs(aj)**2+np.abs(ai)**2-2*ai*np.conj(aj))))
    w = np.real(w)
    if normalized:
        w = w/norm
    return w


def wigner_gkp(s, rl, nr):
    # Compute the Wigner function of the finite-energy GKP
    # sigma should be taken in the range 0.3-1
    mx, mp = grid(rl, nr)
    jtheta = np.vectorize(mpmath.jtheta, 'D')
    C = jtheta(3, 0, np.exp(-4 * math.pi * s ** 2)) * jtheta(3, 0, np.exp(-4 * math.pi * ((s ** 4 + 1) / s ** 2))) \
        + jtheta(2, 0, np.exp(-4 * math.pi * s ** 2)) * jtheta(2, 0, np.exp(-4 * math.pi * ((s ** 4 + 1) / s ** 2)))

    w0 = (1 / (4 * math.pi * np.sqrt(s ** 4 + 1))) * np.exp(-mx ** 2 * (s ** 2 / (s ** 4 + 1))) * np.exp(
        -mp ** 2 * ((s ** 4 + 1) / s ** 2))
    w1 = jtheta(3, (-np.sqrt(math.pi) / 2) * mx / (s ** 4 + 1),
                np.exp(-(math.pi / 4) * (s ** 2 / (s ** 4 + 1)))) * jtheta(3,
                                                                           -1j * (np.sqrt(math.pi) / 2) * mp / s ** 2,
                                                                           np.exp(-math.pi / (4 * s ** 2)))
    w2 = jtheta(3, (-np.sqrt(math.pi) / 2) * mx / (s ** 4 + 1) + math.pi / 2,
                np.exp(-(math.pi / 4) * (s ** 2 / (s ** 4 + 1)))) * jtheta(3, -1j * (
                np.sqrt(math.pi) / 2) * mp / s ** 2 + math.pi / 2, np.exp(-math.pi / (4 * s ** 2)))

    w = C ** (-1) * w0 * (w1 + w2)
    return w


def wigner_particle_in_a_box(a, rl, nr):
    mx, mp = grid(rl, nr)
    mxa = mx[np.abs(mx) < a/2]
    mpa = mp[np.abs(mx) < a/2]
    w = np.zeros([nr, nr])
    w[np.abs(mx) < a/2] = (1/math.pi)*(1-2*(np.abs(mxa)/a))*np.sinc(mpa*(a-2*np.abs(mxa))/math.pi)
    return w


def wigner_gaussian(rl, nr, alpha=0, covmat=np.eye(2) / 2):
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
    w = np.exp(-a / 2) / (2 * math.pi * np.sqrt(np.linalg.det(covmat)))
    return w


def cross_wigner_psi(fi, fj, rl, nr):
    # Wigner function of the projector |fi><fj|
    # fi and fj should be sampled on xl (see get_xl)
    x = np.linspace(-rl / 2, rl / 2, nr)
    wij = np.zeros([nr, nr], dtype=complex)
    for xk in range(nr):
        # fis = np.conj(fi[xk:xk + nr])
        # fjs = np.flip(fj[xk:xk + nr])
        fis = fi[xk:xk + nr]
        fjs = np.conj(np.flip(fj[xk:xk + nr]))
        for pk in range(nr):
            re = scipy.integrate.simpson(np.real(np.exp(-2 * 1j * x[pk] * x) * fis * fjs), x=x) / math.pi
            im = scipy.integrate.simpson(np.imag(np.exp(-2 * 1j * x[pk] * x) * fis * fjs), x=x) / math.pi
            wij[xk][pk] = re + im * 1j
    return wij

def wigner_psi(psi, rl, nr):
    # Wigner function of the wave-function psi
    # psi should be sampled on xl (see get_xl)
    return np.real(cross_wigner_psi(psi, psi, rl, nr))


def density_to_wigner(rho, rl, nr, isherm=True):
    # Build the Wigner function of the density matrix rho (in wij_set basis)
    n = len(rho)
    w = np.zeros([nr, nr])
    if isherm:
        for i in range(n):
            rii = rho[i][i]
            if rii != 0:
                w = w + rii * cross_wigner_fock(i, i, rl, nr)
            for j in range(i+1, n):
                rij = rho[i][j]
                if rij != 0:
                    rho_wij = rho[i][j] * cross_wigner_fock(i, j, rl, nr)
                    w = w + rho_wij + np.conj(rho_wij)
        w = np.real(w)
    else:
        for i in range(n):
            for j in range(n):
                rij = rho[i][j]
                if rij != 0:
                    w = w + rho[i][j] * cross_wigner_fock(i, j, rl, nr)
    return w


def density_to_wigner_from_set(rho, wijset, isherm=True):
    # Build the Wigner function of the density matrix rho (in wij_set basis)
    n = len(rho)
    #nr = np.shape(wijset)[2]
    w = np.zeros(wijset.shape[2:4])
    if isherm:
        for i in range(n):
            w = w + rho[i][i] * wijset[i][i]
            for j in range(i+1, n):
                rho_wij = rho[i][j] * wijset[i][j]
                w = w + 2*np.real(rho_wij)
        w = np.real(w)
    else:
        for i in range(n):
            for j in range(n):
                w = w + rho[i][j] * wijset[i][j]
    return w