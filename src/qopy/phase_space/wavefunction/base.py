import numpy as np
import math
import scipy
from qopy.utils.polynomials import hermite


def psi_fock(n, x):
    # Wave-function of the nth Fock state
    p = np.append(np.zeros(n), 1)
    f = np.multiply(np.exp(-x ** 2 / 2), np.polynomial.hermite.hermval(x, p)) / (math.pi ** (1 / 4) * 2 ** (n / 2) * math.sqrt(scipy.special.factorial(n)))
    return f


def psi_bump(xl, dr=1, pwr=1):
    fx = np.zeros_like(xl)
    dom = np.abs(xl) < (dr/2)
    fx[dom] = np.exp(-pwr*dr**2/(dr**2-(2*xl[dom])**2))
    norm = np.exp(-2*pwr) * (2*pwr)**2 * np.sqrt(np.pi) * scipy.special.hyperu(0.5, 0, 2*pwr)
    norm = norm*dr/2
    return fx/np.sqrt(norm)


def psi_gaussian(x, alpha=0, rsq=0, phi=0):
    # Wave-function of a squeezed coherent state
    # Squeezing is performed with an angle of phi with respect to x-axis, then displaced
    # If alpha is a a tuple: alpha = (Re[alpha]+iIm[alpha])/sqrt(2)
    # <x> = d[0] and <p> = d[1]
    if not (isinstance(alpha, list) or (isinstance(alpha, np.ndarray)) or (isinstance(alpha, tuple))):
        d = np.sqrt(2)*np.array([np.real(alpha), np.imag(alpha)])
    else:
        d = alpha
    x0 = d[0]
    p0 = d[1]

    sigmax2 = np.exp(2 * rsq) * np.sin(phi) ** 2 + np.exp(-2 * rsq) * np.cos(phi) ** 2
    fx = (math.pi * sigmax2) ** (-1/4) \
         * np.exp(-((x - x0) ** 2 / 2)
                  * (np.cosh(rsq) + np.exp(2 * 1j * phi) * np.sinh(rsq))
                  / (np.cosh(rsq) - np.exp(2 * 1j * phi) * np.sinh(rsq))) \
         * np.exp(1j * p0 * x) * np.exp(-1j*x0*p0/2)
    return fx


def psi_gaussian_fock(x, alpha=0, xi=0, n=0):
    # Compute the wave function of a displaced squeezed Fock state, i.e. psi(x) = <x|D(alpha)S(xi)|n>
    # alpha and xi are complex-valued
    # see https://arxiv.org/abs/quant-ph/9612050v1
    x0 = np.sqrt(2)*np.real(alpha)
    p0 = np.sqrt(2)*np.imag(alpha)
    r = np.abs(xi)
    phi = np.angle(xi)
    F1 = np.cosh(r)+np.exp(1j*phi)*np.sinh(r)
    F2 = (1-1j*np.sin(phi)*np.sinh(r)*(np.cosh(r)+np.exp(1j*phi)*np.sinh(r)))/((np.cosh(r)+np.cos(phi)*np.sinh(r))*(np.cosh(r)+np.exp(1j*phi)*np.sinh(r)))
    F3 = (np.cosh(r)+np.exp(-1j*phi)*np.sin(phi)*np.sinh(r))/(np.cosh(r)+np.exp(1j*phi)*np.sin(phi)*np.sinh(r))
    F4 = np.sqrt(np.cosh(r)**2+np.sinh(r)**2+2*np.cos(phi)*np.cosh(r)*np.sinh(r))
    return math.pi**(-1/4)*np.exp(-1j*x0*p0/2)*F1**(-1/2)*np.exp(-((x-x0)**2/2)*F2+1j*p0*x)*(F3**(n/2))*(2**n*math.factorial(n))**(-1/2)*hermite(n, (x - x0) / F4)


def integrate_1d(psi, xl):
    return scipy.integrate.simpson(psi, x=xl)


def normalize_psi(psi, xl):
    norm = integrate_1d(np.abs(psi) ** 2, xl)
    return psi/np.sqrt(norm)


def get_xl(rl, nr):
    # Return the interval over which fx should be sampled to compute its Wigner function
    dx = rl / (2 * (nr - 1))
    drl = (nr / 2) * (rl / (nr - 1))
    xl = np.linspace(-rl / 2 - drl, rl / 2 + drl, 2 * nr) + dx
    return xl


def shannon_entropy_1d(rx, rl):
    # Compute the Shannon entropy (or Rényi entropy) of the density function rx
    nr = len(rx)
    x = np.linspace(-rl / 2, rl / 2, nr)
    rxlog = np.zeros(nr)
    rxlog[np.nonzero(rx)] = np.log(np.abs(rx[np.nonzero(rx)]))
    return - scipy.integrate.simpson(rx * rxlog, x=x)

def renyi_entropy_1d(rx, rl, alpha):
    nr = len(rx)
    x = np.linspace(-rl / 2, rl / 2, nr)
    if alpha == 1:
        return shannon_entropy_1d(rx, rl)
    if math.isinf(alpha):
        m = np.max(np.abs(rx))
        return -np.log(m)
    if alpha == 0:
        rxa = (np.abs(rx) > 0)
        return np.log(scipy.integrate.simpson(rxa, x=x))
    rxa = np.abs(rx) ** alpha
    return (1 / (1 - alpha)) * np.log(scipy.integrate.simpson(rxa, x=x))


def fourier_transform_1d(fx, xl):
    nrl = len(xl)
    phi = np.zeros(nrl, dtype=complex)
    for i in range(nrl):
        pi = xl[i]
        re = scipy.integrate.simpson(np.real(fx*np.exp(-1j*pi*xl)), x=xl)/np.sqrt(2*math.pi)
        im = scipy.integrate.simpson(np.imag(fx*np.exp(-1j*pi*xl)), x=xl)/np.sqrt(2*math.pi)
        phi[i] = re + im * 1j
    return phi


def cubic_phase(gamma, x):
    return np.exp(1j*gamma*x**3)


def ntic_phase(gamma, n, x, abs=False):
    return np.exp(1j*gamma*n*np.abs(x)) if abs else np.exp(1j*gamma*x**n)


def ket_to_psi(ket, x):
    n = len(ket)
    psi = np.zeros(x.shape, dtype=complex)
    for i in range(n):
        psi += ket[i]*psi_fock(i, x)
    return psi


def psi_to_ket(psi, xl, N):
    ket = np.zeros(N, dtype=complex)
    for i in range(N):
        ket[i] = integrate_1d(psi*psi_fock(i, xl), xl)
    return ket
