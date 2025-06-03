import numpy as np
from qopy.utils.grid import grid_square

def fn_xp(alpha1, alpha2, xi1, xi2, x, p):
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


def fn(alpha1, alpha2, xi1, xi2):
    """
    Retourne un callable W(x, p) représentant la fonction de Wigner
    d’un état gaussien croisé : ⟨alpha1, xi1 | W(x, p) | alpha2, xi2⟩
    """

    x01 = np.sqrt(2) * np.real(alpha1)
    p01 = np.sqrt(2) * np.imag(alpha1)
    r1 = np.abs(xi1)
    phi1 = np.angle(xi1)

    x02 = np.sqrt(2) * np.real(alpha2)
    p02 = np.sqrt(2) * np.imag(alpha2)
    r2 = np.abs(xi2)
    phi2 = np.angle(xi2)

    # Fonctions auxiliaires
    def F1(r, phi):
        return np.cosh(r) + np.exp(1j * phi) * np.sinh(r)

    def F2(r, phi):
        num = 1 - 1j * np.sin(phi) * np.sinh(r) * (np.cosh(r) + np.exp(1j * phi) * np.sinh(r))
        den = (np.cosh(r) + np.cos(phi) * np.sinh(r)) * (np.cosh(r) + np.exp(1j * phi) * np.sinh(r))
        return num / den

    F11 = F1(r1, phi1)
    F21 = F2(r1, phi1)
    F12 = F1(r2, phi2)
    F22 = F2(r2, phi2)

    A = (-1/2) * (np.conj(F21) + F22)
    # Préfacteurs indépendants de x,p
    prefact = (
        np.exp(1j * x01 * p01 / 2)
        * np.exp(-1j * x02 * p02 / 2)
        * np.pi ** (-1/2)
        * (np.conj(F11) * F12) ** (-1/2)
        * np.sqrt(-np.pi / A)
    )
    def w(x, p):
        B = 2j * p - 1j * p01 - 1j * p02 - np.conj(F21) * (x - x01) + F22 * (x - x02)
        C = (
            -1j * p01 * x
            + 1j * p02 * x
            - 0.5 * np.conj(F21) * (x - x01) ** 2
            - 0.5 * F22 * (x - x02) ** 2
        )
        return (1 / np.pi) * prefact * np.exp(-B**2 / (4 * A) + C)
    return w


def grid(alpha1, alpha2, xi1, xi2, rl, nr):
    mx, mp = grid_square(rl, nr)
    wij = fn_xp(alpha1, alpha2, xi1, xi2, mx, mp)
    return wij