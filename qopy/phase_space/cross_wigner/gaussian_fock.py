import numpy as np
import scipy
import qopy.utils.polynomials as poly
from qopy.utils.grid import grid_square

def fn_xp(alpha1, alpha2, xi1, xi2, n1, n2, x, p):
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


def grid(alpha1, alpha2, xi1, xi2, n1, n2, rl, nr):
    mx, mp = grid_square(rl, nr)
    return fn_xp(alpha1, alpha2, xi1, xi2, n1, n2, mx, mp)


def fn(alpha1, alpha2, xi1, xi2, n1, n2):
    """
    Retourne une fonction W(x, p) pour la Wigner function d’un état de Fock
    squeezé-déplacé croisé : ⟨alpha1, xi1, n1 | W(x, p) | alpha2, xi2, n2⟩
    """

    def F1(r, phi):
        return np.cosh(r) + np.exp(1j * phi) * np.sinh(r)
    def F2(r, phi):
        num = 1 - 1j * np.sin(phi) * np.sinh(r) * (np.cosh(r) + np.exp(1j * phi) * np.sinh(r))
        den = (np.cosh(r) + np.cos(phi) * np.sinh(r)) * (np.cosh(r) + np.exp(1j * phi) * np.sinh(r))
        return num / den
    def F3(r, phi):
        num = np.cosh(r) + np.exp(-1j * phi) * np.sin(phi) * np.sinh(r)
        den = np.cosh(r) + np.exp(1j * phi) * np.sin(phi) * np.sinh(r)
        return num / den
    def F4(r, phi):
        return np.sqrt(np.cosh(r)**2 + np.sinh(r)**2 + 2 * np.cos(phi) * np.cosh(r) * np.sinh(r))

    x01 = np.sqrt(2) * np.real(alpha1)
    p01 = np.sqrt(2) * np.imag(alpha1)
    r1 = np.abs(xi1)
    phi1 = np.angle(xi1)

    F11 = F1(r1, phi1)
    F21 = F2(r1, phi1)
    F31 = F3(r1, phi1)
    F41 = F4(r1, phi1)

    x02 = np.sqrt(2) * np.real(alpha2)
    p02 = np.sqrt(2) * np.imag(alpha2)
    r2 = np.abs(xi2)
    phi2 = np.angle(xi2)

    F12 = F1(r2, phi2)
    F22 = F2(r2, phi2)
    F32 = F3(r2, phi2)
    F42 = F4(r2, phi2)

    A = (-1 / 2) * (np.conj(F21) + F22)
    f = -A
    a = -2 / F42
    c = 2 / F41

    norm = (
        np.exp(1j * x01 * p01 / 2)
        * np.exp(-1j * x02 * p02 / 2)
        * np.pi ** (-1 / 2)
        * (np.conj(F11) * F12) ** (-1 / 2)
        * np.conj(F31) ** (n1 / 2)
        * F32 ** (n2 / 2)
        * (2 ** (n1 + n2) * scipy.special.factorial(n1) * scipy.special.factorial(n2)) ** (-1 / 2)
    )
    def w(x, p):
        alpha = (
            2j * p
            - 1j * p01 - 1j * p02
            - np.conj(F21) * (x - x01)
            + F22 * (x - x02)
        )
        C = (
            -1j * p01 * x + 1j * p02 * x
            - 0.5 * np.conj(F21) * (x - x01) ** 2
            - 0.5 * F22 * (x - x02) ** 2
        )
        b = 2 * (x - x02) / F42
        d = 2 * (x - x01) / F41

        integral = np.sqrt(np.pi / f) * np.exp(alpha**2 / (4 * f)) * poly.hermite_2index(
            n2, n1,
            b + a * alpha / (2 * f), -1 + a**2 / (4 * f),
            d + c * alpha / (2 * f), -1 + c**2 / (4 * f),
            a * c / (2 * f)
        )
        return np.nan_to_num((1 / np.pi) * norm * np.exp(C) * integral)
    return w