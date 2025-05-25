'''

Contains all the functions creating density operators, or acting on density operators

'''



import numpy as np
import math
import scipy
import random



def rho_rotate(rho, phi):
    N = len(rho)
    rotmat = np.diag(np.exp(-1j*phi*np.arange(N)))
    rhorot = rotmat @ rho @ np.conj(rotmat)
    return rhorot


def ket_to_rho(ket):
    # Build rho from the vector ket
    ket = np.array(ket)
    if ket.size == 1:
        rho = np.ones([1, 1])*np.abs(ket)**2
        return rho
    n = len(ket)
    rho = np.zeros([n, n], dtype=complex)
    for i in range(n):
        rho[i][i] = np.abs(ket[i]) ** 2
        for j in range(i + 1, n):
            rij = ket[i] * np.conj(ket[j])
            rho[i][j] = rij
            rho[j][i] = np.conj(rij)
    return rho



def rho_phase(rho, phi):
    n = len(rho)
    nlist = np.arange(0, n)
    uphi = np.diag(np.exp(-1j*phi*nlist))
    return uphi @ rho @ np.conj(np.transpose(uphi))


def rho_trim(rho, tol=0):
    n = len(rho)
    iszero = True
    for i in range(n - 1):
        iszero = (iszero and (np.abs(rho[n - 1][i])) <= tol)
        iszero = (iszero and (np.abs(rho[i][n - 1])) <= tol)
    iszero = (iszero and (np.abs(rho[n - 1][n - 1])) <= tol)
    if iszero:
        rho = rho[:n - 1, :n - 1]
        if len(rho) > 1:
            rho = rho_trim(rho, tol)
    return rho


def rand_ket(n, form='rho'):
    if n == 1:
        if form == 'rho':
            return np.array([[1]])
        return np.array([np.exp(1j*random.random()*2*math.pi)])
    ket = random_unitary(n)[0]
    if form == 'rho':
        return ket_to_rho(ket)
    return ket


def density_random(n):
    if n == 1:
        return np.ones([1, 1])
    u_rand1 = scipy.stats.unitary_group.rvs(n)
    u_rand2 = scipy.stats.unitary_group.rvs(n)
    rho_diag = np.diag(np.abs(u_rand1[0])**2)
    rho = u_rand2 @ rho_diag @ np.transpose(np.conj(u_rand2))
    return rho


def random_unitary(n):
    if n == 1:
        return np.array([np.exp(1j*random.random()*2*math.pi)])
    return scipy.stats.unitary_group.rvs(n)


def displacement_matrix(m, n, alpha):
    # return the value of <m|D(alpha)|n>
    dmn = 0
    for p in range(np.min([m, n])+1):
        dmn = dmn + np.sqrt(float(math.factorial(m)*math.factorial(n)))*(-1)**(n-p)/(math.factorial(p)*math.factorial(m-p)*math.factorial(n-p))*alpha**(m-p)*np.conj(alpha)**(n-p)
    dmn = np.exp(-(1/2)*np.abs(alpha)**2)*dmn
    return dmn


def density_bloch_sphere(theta, phi=0, p=1):
    ket = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
    rho_pure = ket_to_rho(ket)
    rho_mix = p*rho_pure+(1-p)*np.eye(2)/2
    return rho_mix