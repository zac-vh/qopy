import numpy as np
from qopy.utils.grid import grid_square
import scipy
import qopy.utils.polynomials as poly


def coherent_xp(alpha, beta, x, p):
    re = (alpha+np.conj(beta))/2
    im = (alpha-np.conj(beta))/(2*1j)
    exp = -(1/2)*np.abs(alpha)**2-(1/2)*np.abs(beta)**2+alpha*np.conj(beta)\
          -(x-np.sqrt(2)*re)**2-(p-np.sqrt(2)*im)**2
    return (1/np.pi)*np.exp(exp)

def coherent(alpha, beta, rl, nr):
    mx, mp = grid_square(rl, nr)
    wij = coherent_xp(alpha, beta, mx, mp)
    return wij


def displaced_fock_xp(m, n, alpha, beta, x, p):
    # Return the Wigner function of the operator D(alpha)|m><n|D(-beta)
    gamma = np.sqrt(2)*(x+1j*p)
    delta = gamma-alpha-beta
    sum = 0
    for p in range(min(m, n)+1):
        sum = sum + (-1)**p/(scipy.special.factorial(p)*scipy.special.factorial(n-p)*scipy.special.factorial(m-p))*delta**(n-p)*np.conj(delta)**(m-p)

    exp = np.exp(-(1/2)*(np.abs(gamma)**2+np.abs(alpha)**2+np.abs(beta)**2))*np.exp(-alpha*np.conj(beta)+gamma*np.conj(beta)+alpha*np.conj(gamma))
    return (np.sqrt(scipy.special.factorial(m)*scipy.special.factorial(n))/np.pi)*exp*sum


def displaced_fock(m, n, alpha, beta, rl, nr):
    # Return the Wigner function of the operator D(alpha)|m><n|D(-beta)
    mx, mp = grid_square(rl, nr)
    wij = displaced_fock_xp(m, n, alpha, beta, mx, mp)
    return wij


def fock_xp(i, j, x, p):
    # Return the Wigner function of |i><j| evaluated at (x,p)
    # x, p should have the same dimension
    r = np.sqrt(x ** 2 + p ** 2)
    if j == i:
        n = i
        w = (-1) ** n * (1 / np.pi) * np.exp(-r ** 2)*scipy.special.laguerre(n)(2 * r ** 2)
        return w
    if i < j:
        m, n = i, j
        w = (1 / np.pi) * np.exp(-r ** 2) * (-1) ** m * np.sqrt(2) ** (n - m) \
            * np.sqrt(scipy.special.factorial(m) / scipy.special.factorial(n)) * \
            (x + 1j * p) ** (n - m) * scipy.special.genlaguerre(m, n - m)(2 * r ** 2)
        return w
    if i > j:
        m, n = j, i
        w = (1 / np.pi) * np.exp(-r ** 2) * (-1) ** m * np.sqrt(2) ** (n - m) \
            * np.sqrt(scipy.special.factorial(m) / scipy.special.factorial(n)) * \
            (x - 1j * p) ** (n - m) * scipy.special.genlaguerre(m, n - m)(2 * r ** 2)
        return w


def fock(i, j, rl, nr):
    mx, mp = grid_square(rl, nr)
    wij = fock_xp(i, j, mx, mp)
    return wij


def fock_set(N, rl, nr):
    wijset = np.empty([N, N, nr, nr], dtype=complex)
    for i in range(N):
        wijset[i, i] = fock(i, i, rl, nr)
        for j in range(i + 1, N):
            wij = fock(i, j, rl, nr)
            wijset[i, j] = wij
            wijset[j, i] = np.conj(wij)
    return wijset


def from_psi(f, g, rl, nr):
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
        wij[xk, :] = dx * np.dot(phase, integrand) / np.pi  # dot over axis=1

    return wij


def gaussian_xp(alpha1, alpha2, xi1, xi2, x, p):
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
    prefact = np.exp(1j*x01*p01/2)*np.exp(-1j*x02*p02/2)*np.pi**(-1/2)*(np.conj(F11)*F12)**(-1/2)
    w = (1/np.pi)*prefact*np.sqrt(-np.pi/A)*np.exp(-B**2/(4*A)+C)
    return w


def gaussian(alpha1, alpha2, xi1, xi2, rl, nr):
    mx, mp = grid_square(rl, nr)
    wij = gaussian_xp(alpha1, alpha2, xi1, xi2, mx, mp)
    return wij


def gaussian_fock_xp(alpha1, alpha2, xi1, xi2, n1, n2, x, p):
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

    prefact = np.exp(1j*x01*p01/2)*np.exp(-1j*x02*p02/2)*np.pi **(-1/2)*(np.conj(F11)*F12)**(-1/2)*np.conj(F31)**(n1/2)*F32**(n2/2)*(2**(n1+n2)*scipy.special.factorial(n1)*scipy.special.factorial(n2))**(-1/2)*np.exp(C)
    integral = np.sqrt(np.pi/f)*np.exp(alpha**2/(4*f))*poly.hermite_2index(n2, n1, b+a*alpha/(2*f), -1+a**2/(4*f), d+c*alpha/(2*f), -1+c**2/(4*f), a*c/(2*f))
    return np.nan_to_num(prefact * integral / np.pi)


def gaussian_fock(alpha1, alpha2, xi1, xi2, n1, n2, rl, nr):
    mx, mp = grid_square(rl, nr)
    return gaussian_fock_xp(alpha1, alpha2, xi1, xi2, n1, n2, mx, mp)