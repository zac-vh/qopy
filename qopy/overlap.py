import numpy as np
import scipy


def coherent(alpha, beta):
    # Computes <alpha|beta>
    return np.exp(-(1 / 2) * (np.abs(alpha) ** 2 + np.abs(beta) ** 2 - 2 * beta * np.conj(alpha)))


def gaussian(alpha1, alpha2, xi1, xi2):
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


def gaussian_fock(alpha1, alpha2, xi1, xi2, n1, n2):
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
