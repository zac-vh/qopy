'''

Contains all the functions creating density operators, or acting on density operators

'''



import numpy as np
import math
import scipy
import random



def rotate(rho, phi):
    rotmat = np.diag(np.exp(-1j*phi*np.arange(len(rho))))
    return rotmat @ rho @ np.conj(rotmat)


def from_ket(ket):
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


def trim(rho, tol=0):
    n = len(rho)
    iszero = True
    for i in range(n - 1):
        iszero = (iszero and (np.abs(rho[n - 1][i])) <= tol)
        iszero = (iszero and (np.abs(rho[i][n - 1])) <= tol)
    iszero = (iszero and (np.abs(rho[n - 1][n - 1])) <= tol)
    if iszero:
        rho = rho[:n - 1, :n - 1]
        if len(rho) > 1:
            rho = trim(rho, tol)
    return rho


def random_pure(N):
    if N == 1:
        return np.array([[1]])
    return from_ket(random_unitary(N)[0])


def random_mixed(N):
    if N == 1:
        return np.array([[1]])
    u_rand1 = scipy.stats.unitary_group.rvs(N)
    u_rand2 = scipy.stats.unitary_group.rvs(N)
    rho_diag = np.diag(np.abs(u_rand1[0])**2)
    rho = u_rand2 @ rho_diag @ np.transpose(np.conj(u_rand2))
    return rho


def random_unitary(N):
    if N == 1:
        return np.array([np.exp(1j*random.random()*2*math.pi)])
    return scipy.stats.unitary_group.rvs(N)


def displacement_matrix(m, n, alpha):
    # return the value of <m|D(alpha)|n>
    dmn = 0
    for p in range(np.min([m, n])+1):
        dmn = dmn + np.sqrt(float(math.factorial(m)*math.factorial(n)))*(-1)**(n-p)/(math.factorial(p)*math.factorial(m-p)*math.factorial(n-p))*alpha**(m-p)*np.conj(alpha)**(n-p)
    dmn = np.exp(-(1/2)*np.abs(alpha)**2)*dmn
    return dmn


def rho_bloch_sphere(theta, phi=0, p=1):
    ket = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
    rho_pure = from_ket(ket)
    rho_mix = p*rho_pure+(1-p)*np.eye(2)/2
    return rho_mix