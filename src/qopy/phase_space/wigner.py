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
from qopy.utils.linalg import matrix_eigenvalues_eigenvectors
from qopy.phase_space.wavefunction import ket_to_psi
from qopy.phase_space.wavefunction import get_xl
import qopy.utils.polynomials as poly


def wigner_radial_fock(n, r):
    # Radial function of the Wigner function of the nth Fock state
    p = np.append(np.zeros(n), 1)
    wr = (-1) ** n * (1 / math.pi) * np.multiply(np.exp(-r ** 2), np.polynomial.laguerre.lagval(2 * r ** 2, p))
    return wr


def wigner_area_fock(n, a):
    # Area function associated to the Wigner function of the nth Fock state
    # F(a) = pi*W(sqrt(a))
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


def density_to_wigner(rho, rl, nr, isherm=True):
    n = rho.shape[0]
    w = np.zeros((nr, nr), dtype=complex)
    for i in range(n):
        for j in range(i+1 if isherm else n):
            rij = rho[i, j]
            if rij == 0:
                continue
            wij = cross_wigner_fock(i, j, rl, nr)
            if isherm:
                if i != j:
                    w += 2*np.real(rij * wij)
                if i == j:
                    w += rij * wij
            else:
                w += rij * wij
    return np.real(w) if isherm else w


def density_to_wigner_from_set(rho, wijset, isherm=True):
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


def density_to_wigner_from_psi(rho, rl, nr):
    # Compute the Wigner function from the wavefunction of the eigenvectors of rho
    # More efficient when N >> 1
    xl = get_xl(rl, nr)
    w = np.zeros((nr, nr), dtype=complex)
    eigvals, eigkets = matrix_eigenvalues_eigenvectors(rho)
    N = len(eigvals)
    for j in range(N):
        psij = ket_to_psi(eigkets[j], xl)
        w += eigvals[j] * wigner_psi(psij, rl, nr)
    return np.real(w)


def cross_wigner_fock_set(N, rl, nr):
    wijset = np.empty([N, N, nr, nr], dtype=complex)
    for i in range(N):
        wijset[i, i] = wigner_fock(i, rl, nr)
        for j in range(i + 1, N):
            wij = cross_wigner_fock(i, j, rl, nr)
            wijset[i, j] = wij
            wijset[j, i] = np.conj(wij)
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


def cross_wigner_coherent(alpha, beta, rl, nr, sq=1):
    mx, mp = grid(rl, nr)
    wij = cross_wigner_coherent_xp(alpha, beta, sq * mx, mp / sq)
    return wij


def wigner_coherent_superposition(alphas, cis, rl, nr, sq=1, normalized=True):
    w = np.zeros([nr, nr])
    n = len(alphas)
    norm = 0
    for i in range(n):
        ai = alphas[i]
        ci = cis[i]
        w = w + np.abs(ci) ** 2 * cross_wigner_coherent(ai, ai, rl, nr, sq)
        norm = norm + np.abs(ci)**2
        for j in range(i+1, n):
            aj = alphas[j]
            cj = cis[j]
            w = w + 2*np.real(ci * np.conj(cj) * cross_wigner_coherent(ai, aj, rl, nr, sq))
            norm = norm + 2*np.real(np.conj(cj)*ci*np.exp(-(1/2)*(np.abs(aj)**2+np.abs(ai)**2-2*ai*np.conj(aj))))
    w = np.real(w)
    if normalized:
        w = w/norm
    return w


def cross_wigner_gaussian_xp(alpha1, alpha2, xi1, xi2, x, p):
    x01 = np.sqrt(2) * np.real(alpha1)
    p01 = np.sqrt(2) * np.imag(alpha1)
    r1 = np.abs(xi1)
    phi1 = np.angle(xi1)
    x02 = np.sqrt(2) * np.real(alpha2)
    p02 = np.sqrt(2) * np.imag(alpha2)
    r2 = np.abs(xi2)
    phi2 = np.angle(xi2)
    F11 = np.cosh(r1) + np.exp(1j * phi1) * np.sinh(r1)
    F21 = (1 - 1j * np.sin(phi1) * np.sinh(r1) * (np.cosh(r1) + np.exp(1j * phi1) * np.sinh(r1))) / ((np.cosh(r1) + np.cos(phi1) * np.sinh(r1)) * (np.cosh(r1) + np.exp(1j * phi1) * np.sinh(r1)))
    F12 = np.cosh(r2) + np.exp(1j * phi2) * np.sinh(r2)
    F22 = (1 - 1j * np.sin(phi2) * np.sinh(r2) * (np.cosh(r2) + np.exp(1j * phi2) * np.sinh(r2))) / ((np.cosh(r2) + np.cos(phi2) * np.sinh(r2)) * (np.cosh(r2) + np.exp(1j * phi2) * np.sinh(r2)))
    A = (-1/2)*(np.conj(F21)+F22)
    B = 2*1j*p-1j*p01-1j*p02-np.conj(F21)*(x-x01)+F22*(x-x02)
    C = -1j*p01*x+1j*p02*x-(1/2)*np.conj(F21)*(x-x01)**2-(1/2)*F22*(x-x02)**2
    prefact = np.exp(1j*x01*p01/2)*np.exp(-1j*x02*p02/2)*math.pi**(-1/2)*(np.conj(F11)*F12)**(-1/2)
    w = (1/math.pi)*prefact*np.sqrt(-math.pi/A)*np.exp(-B**2/(4*A)+C)
    return w


def cross_wigner_gaussian(alpha1, alpha2, xi1, xi2, rl, nr):
    mx, mp = grid(rl, nr)
    wij = cross_wigner_gaussian_xp(alpha1, alpha2, xi1, xi2, mx, mp)
    return wij


def overlap_gaussian(alpha1, alpha2, xi1, xi2):
    #see https://doi.org/10.1103/PhysRevA.54.5378
    #return <0|S(xi1)D(alpha1)D(alpha2)S(xi2)|0>
    r1 = np.abs(xi1)
    r2 = np.abs(xi2)
    phi1 = np.angle(xi1)
    phi2 = np.angle(xi2)
    sigma21 = np.cosh(r1)*np.cosh(r2)-np.exp(1j*(phi2-phi1))*np.sinh(r1)*np.sinh(r2)
    eta21 = (alpha2-alpha1)*np.cosh(r2)-(np.conj(alpha2)-np.conj(alpha1))*np.exp(1j*phi2)*np.sinh(r2)
    eta12 = (alpha1-alpha2)*np.cosh(r1)-(np.conj(alpha1)-np.conj(alpha2))*np.exp(1j*phi1)*np.sinh(r1)
    return np.exp((eta21*np.conj(eta12)/(2*sigma21))+(1/2)*(alpha2*np.conj(alpha1)-np.conj(alpha2)*alpha1))/np.sqrt(sigma21)


def wigner_gaussian_superposition(alpha_list, xi_list, amplitude_list, rl, nr, normalized=True):
    w = np.zeros([nr, nr])
    n = len(alpha_list)
    norm = 0
    for i in range(n):
        ai = alpha_list[i]
        xi = xi_list[i]
        ci = amplitude_list[i]
        w = w + np.abs(ci) ** 2 * cross_wigner_gaussian(ai, ai, xi, xi, rl, nr)
        norm = norm + np.abs(ci)**2
        for j in range(i+1, n):
            aj = alpha_list[j]
            xj = xi_list[j]
            cj = amplitude_list[j]
            w = w + 2*np.real(ci * np.conj(cj) * cross_wigner_gaussian(ai, aj, xi, xj, rl, nr))
            norm = norm + 2*np.real(np.conj(cj)*ci*overlap_gaussian(ai, aj, xi, xj))
    w = np.real(w)
    if normalized:
        w = w/norm
    return w


def cross_wigner_gaussian_fock_xp(alpha1, alpha2, xi1, xi2, n1, n2, x, p):
    # References:
    # Wave function of a displaced squeezed Fock state: https://doi.org/10.1016/S0375-9601(97)00183-7
    # Integral of Hermite polynomials: https://doi.org/10.1016/j.aml.2012.02.043
    # Overlap of displaced squeezed Fock states: https://doi.org/10.1103/PhysRevA.54.5378

    x01 = np.sqrt(2)*np.real(alpha1)
    p01 = np.sqrt(2)*np.imag(alpha1)
    r1 = np.abs(xi1)
    phi1 = np.angle(xi1)

    F11 = np.cosh(r1)+np.exp(1j*phi1)*np.sinh(r1)
    F21 = (1-1j*np.sin(phi1)*np.sinh(r1)*(np.cosh(r1)+np.exp(1j*phi1)*np.sinh(r1)))/((np.cosh(r1)+np.cos(phi1)*np.sinh(r1))*(np.cosh(r1)+np.exp(1j*phi1)*np.sinh(r1)))
    F31 = (np.cosh(r1)+np.exp(-1j*phi1)*np.sin(phi1)*np.sinh(r1))/(np.cosh(r1)+np.exp(1j*phi1)*np.sin(phi1)*np.sinh(r1))
    F41 = np.sqrt(np.cosh(r1)**2+np.sinh(r1)**2+2*np.cos(phi1)*np.cosh(r1)*np.sinh(r1))

    x02 = np.sqrt(2)*np.real(alpha2)
    p02 = np.sqrt(2)*np.imag(alpha2)
    r2 = np.abs(xi2)
    phi2 = np.angle(xi2)

    F12 = np.cosh(r2)+np.exp(1j*phi2)*np.sinh(r2)
    F22 = (1-1j*np.sin(phi2)*np.sinh(r2)*(np.cosh(r2)+np.exp(1j*phi2)*np.sinh(r2)))/((np.cosh(r2)+np.cos(phi2)*np.sinh(r2))*(np.cosh(r2)+np.exp(1j*phi2)*np.sinh(r2)))
    F32 = (np.cosh(r2)+np.exp(-1j*phi2)*np.sin(phi2)*np.sinh(r2))/(np.cosh(r2)+np.exp(1j*phi2)*np.sin(phi2)*np.sinh(r2))
    F42 = np.sqrt(np.cosh(r2)**2+np.sinh(r2)**2+2*np.cos(phi2)*np.cosh(r2)*np.sinh(r2))

    A = (-1/2)*(np.conj(F21)+F22)
    B = 2*1j*p-1j*p01-1j*p02-np.conj(F21)*(x-x01)+F22*(x-x02)
    C = -1j*p01*x+1j*p02*x-(1/2)*np.conj(F21)*(x-x01)**2-(1/2)*F22*(x-x02)**2

    f = -A
    alpha = B
    a = -2/F42
    b = 2*(x-x02)/F42
    c = 2/F41
    d = 2*(x-x01)/F41

    prefact = np.exp(1j*x01*p01/2)*np.exp(-1j*x02*p02/2)*math.pi **(-1/2)*(np.conj(F11)*F12)**(-1/2)*np.conj(F31)**(n1/2)*F32**(n2/2)*(2**(n1+n2)*scipy.special.factorial(n1)*scipy.special.factorial(n2))**(-1/2)*np.exp(C)
    integral = np.sqrt(np.pi/f)*np.exp(alpha**2/(4*f))*poly.hermite_2index(n2, n1, b+a*alpha/(2*f), -1+a**2/(4*f), d+c*alpha/(2*f), -1+c**2/(4*f), a*c/(2*f))
    return np.nan_to_num(prefact * integral / np.pi)


def cross_wigner_gaussian_fock(alpha1, alpha2, xi1, xi2, n1, n2, rl, nr):
    mx, mp = grid(rl, nr)
    return cross_wigner_gaussian_fock_xp(alpha1, alpha2, xi1, xi2, n1, n2, mx, mp)


def overlap_gaussian_fock(alpha1=0, alpha2=0, xi1=0, xi2=0, n1=0, n2=0):
    # see https://doi.org/10.1103/PhysRevA.54.5378
    # return <n1|S^dag(xi1)D^dag(alpha1)D(alpha2)S(xi2)|n2>

    r1 = np.abs(xi1)
    r2 = np.abs(xi2)
    phi1 = np.angle(xi1)
    phi2 = np.angle(xi2)

    sigma21 = np.cosh(r2)*np.cosh(r1)-np.exp(1j*(phi2-phi1))*np.sinh(r2)*np.sinh(r1)
    eta21 = (alpha2-alpha1)*np.cosh(r2)-(np.conj(alpha2)-np.conj(alpha1))*np.exp(1j*phi2)*np.sinh(r2)
    eta12 = (alpha1-alpha2)*np.cosh(r1)-(np.conj(alpha1)-np.conj(alpha2))*np.exp(1j*phi1)*np.sinh(r1)
    delta21 = np.exp(1j*phi1)*np.sinh(r1)*np.cosh(r2)-np.exp(1j*phi2)*np.sinh(r2)*np.cosh(r1)

    # Handling Gaussian case
    if n1 == 0 and n2 == 0:
        return np.exp((eta21*np.conj(eta12)/(2*sigma21))+(1/2)*(alpha2*np.conj(alpha1)-np.conj(alpha2)*alpha1))/np.sqrt(sigma21)

    # Handling the case eta12 = eta 12 = 0
    if alpha1 == alpha2:
        if (n1 % 2) != (n2 % 2):
            return 0
        res = 0
        for l in range(n1 % 2, min(n1, n2)+1, 2):
            j = (n1 - l) / 2
            k = (n2 - l) / 2
            res += np.sqrt(scipy.special.factorial(n1)*scipy.special.factorial(n2))/(scipy.special.factorial(j)*scipy.special.factorial(k)*scipy.special.factorial(l))*(-delta21*sigma21/2)**j*(np.conj(delta21)*sigma21/2)**k*sigma21**l
        return res/np.sqrt(sigma21)

    # General case
    factj = -delta21*sigma21/(2*eta21**2)
    factk = np.conj(delta21)*sigma21/(2*np.conj(eta12)**2)
    factl = sigma21/(eta21*np.conj(eta12))
    res = 0
    for j in range(int(n1/2)+1):
        for k in range(int(n2/2)+1):
            m1 = n1-2*j
            m2 = n2-2*k
            for l in range(min(m1, m2)+1):
                res += np.sqrt(scipy.special.factorial(n1)*scipy.special.factorial(n2))/(scipy.special.factorial(j)*scipy.special.factorial(k)*scipy.special.factorial(l)*scipy.special.factorial(m1-l)*scipy.special.factorial(m2-l))*factj**j*factk**k*factl**l
    return res*np.exp(((eta21*np.conj(eta12)/(2*sigma21)))+(1/2)*(alpha2*np.conj(alpha1)-np.conj(alpha2)*alpha1))*(eta21/sigma21)**n1*(np.conj(eta12)/sigma21)**n2/np.sqrt(sigma21)


def wigner_gaussian_fock_superposition(alpha_list, xi_list, n_list, amplitude_list, rl, nr, normalized=True):
    w = np.zeros([nr, nr])
    n = len(amplitude_list)
    norm = 0
    for i in range(n):
        ai = alpha_list[i]
        xi = xi_list[i]
        ni = n_list[i]
        ci = amplitude_list[i]
        w = w + np.abs(ci) ** 2 * cross_wigner_gaussian_fock(ai, ai, xi, xi, ni, ni, rl, nr)
        norm = norm + np.abs(ci)**2
        for j in range(i+1, n):
            aj = alpha_list[j]
            xj = xi_list[j]
            nj = n_list[j]
            cj = amplitude_list[j]
            w = w + 2*np.real(ci * np.conj(cj) * cross_wigner_gaussian_fock(ai, aj, xi, xj, ni, nj, rl, nr))
            norm = norm + 2*np.real(np.conj(cj)*ci*overlap_gaussian_fock(ai, aj, xi, xj, ni, nj))
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


def alpha_to_displacement(alpha):
    return np.sqrt(2)*np.array([np.real(alpha), np.imag(alpha)])


def wigner_gaussian(rl, nr, alpha=0, covmat=np.eye(2) / 2):
    # Wigner function of a Gaussian state with mean d and covariance matrix gamma
    if not (isinstance(alpha, list) or (isinstance(alpha, np.ndarray)) or (isinstance(alpha, tuple))):
        d = alpha_to_displacement(alpha)
    else:
        d = alpha
    [g11, g12], [g21, g22] = np.linalg.inv(covmat)
    dx, dy = d
    x = np.linspace(-rl / 2, rl / 2, nr)
    [mx, my] = np.meshgrid(x, x, indexing='ij')
    a = g11 * (mx - dx) ** 2 + (g21 + g12) * (mx - dx) * (my - dy) + g22 * (my - dy) ** 2
    w = np.exp(-a / 2) / (2 * math.pi * np.sqrt(np.linalg.det(covmat)))
    return w


def cross_wigner_psi(f, g, rl, nr):
    """
    Fast version: vectorized over p, avoids double for-loop and simpson.
    """
    x = np.linspace(-rl / 2, rl / 2, nr)
    dx = x[1] - x[0]
    wij = np.zeros((nr, nr), dtype=complex)

    phase = np.exp(2j * np.outer(x, x))  # shape (nr, nr)

    for xk in range(nr):
        fk = np.conj(f[xk:xk + nr])       # shape (nr,)
        gk = np.flip(g[xk:xk + nr])       # shape (nr,)
        integrand = fk * gk               # shape (nr,)
        wij[xk, :] = dx * np.dot(phase, integrand) / math.pi  # dot over axis=1

    return wij



def wigner_psi(psi, rl, nr):
    # Wigner function of the wave-function psi
    # psi should be sampled on xl (see get_xl)
    return np.real(cross_wigner_psi(psi, psi, rl, nr))


def wigner_cubic_phase(gamma, rl, nr, disp=(0, 0), sq=1):
    # Compute the Wigner function of a cubic-phase-gate acting on vacuum
    if gamma == 0:
        return wigner_fock(0, rl, nr)
    mx, mp = grid(rl, nr)
    mx = (mx-disp[0])/sq
    mp = (mp-disp[1])*sq
    mgamma = np.exp(1/(54*gamma**2))*(4/(3*np.abs(gamma)))**(1/3)/np.sqrt(math.pi)
    airy = scipy.special.airy((4/(3*gamma))**(1/3)*(3*gamma*mx**2-mp+1/(12*gamma)))[0]
    wcubic = mgamma*np.exp(-mp/(3*gamma))*airy
    return wcubic
