import numpy as np
import scipy
from qopy.utils.grid import grid_square


def fn_xp(m, n, alpha, beta, x, p):
    # Return the Wigner function of the operator D(alpha)|m><n|D(-beta)
    gamma = np.sqrt(2)*(x+1j*p)
    delta = gamma-alpha-beta
    sum = 0
    for p in range(min(m, n)+1):
        sum = sum + (-1)**p/(scipy.special.factorial(p)*scipy.special.factorial(n-p)*scipy.special.factorial(m-p))*delta**(n-p)*np.conj(delta)**(m-p)

    exp = np.exp(-(1/2)*(np.abs(gamma)**2+np.abs(alpha)**2+np.abs(beta)**2))*np.exp(-alpha*np.conj(beta)+gamma*np.conj(beta)+alpha*np.conj(gamma))
    return (np.sqrt(scipy.special.factorial(m)*scipy.special.factorial(n))/np.pi)*exp*sum


def fn(m, n, alpha, beta):
    """
    Returns a callable Wigner function corresponding to D(alpha)|m><n|D(-beta),
    optimized for evaluation on numpy meshgrids (x, p).
    """
    # Precompute constants
    fact = np.sqrt(scipy.special.factorial(m) * scipy.special.factorial(n)) / np.pi
    exp_prefactor = np.exp(-0.5 * (np.abs(alpha)**2 + np.abs(beta)**2) - alpha * np.conj(beta))
    def w(x, p):
        gamma = np.sqrt(2) * (x + 1j * p)
        delta = gamma - alpha - beta
        # Compute the sum over p from 0 to min(m,n)
        minmn = min(m, n)
        result = np.zeros_like(x, dtype=complex)
        for k in range(minmn + 1):
            coeff = (-1)**k / (scipy.special.factorial(k) * scipy.special.factorial(n - k) * scipy.special.factorial(m - k))
            term = coeff * (delta ** (n - k)) * (np.conj(delta) ** (m - k))
            result += term
        full_exp = np.exp(np.real(gamma * np.conj(beta) + alpha * np.conj(gamma) - np.abs(gamma)**2 / 2))
        return fact * exp_prefactor * result * full_exp
    return w


def grid(m, n, alpha, beta, rl, nr):
    # Return the Wigner function of the operator D(alpha)|m><n|D(-beta)
    mx, mp = grid_square(rl, nr)
    wij = fn_xp(m, n, alpha, beta, mx, mp)
    return wij