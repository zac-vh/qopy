'''

Contains all the functions creating density operators, or acting on density operators

'''



import numpy as np
import math





def rho_rotate(rho, phi):
    N = len(rho)
    rotmat = np.diag(np.exp(-1j*phi*np.arange(N)))
    rhorot = rotmat @ rho @ np.conj(rotmat)
    return rhorot





def rho_covmat(rho, mean=None):
    n = len(rho)
    if mean is None:
        [xm, pm] = rho_mean(rho)
    elif not (isinstance(mean, list) or (isinstance(mean, np.ndarray)) or (isinstance(mean, tuple))):
        xm = np.sqrt(2)*np.real(mean)
        pm = np.sqrt(2)*np.imag(mean)
    else:
        xm = mean[0]
        pm = mean[1]

    nm = rho_s(rho, 'photons')
    aa = 0+0j
    for i in range(n-2):
        aa = aa + rho[i, i+2]*np.sqrt((i+1)*(i+2))
    xxm = np.real(aa)+nm+1/2-xm**2
    ppm = -np.real(aa)+nm+1/2-pm**2
    xpm = -np.imag(aa)-xm*pm
    cov = np.array([[xxm, xpm], [xpm, ppm]])
    return cov


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


def rho_s(rho, a=1):
    # Compute the Von Neumann entropy (or Rényi entropy) of rho
    # Eigenvalues of rho are assumed positive
    la = np.real(np.linalg.eig(rho)[0])
    la = np.delete(la, np.where(la <= 0))
    if a == 'purity':
        return np.sum(la**2)
    if a == 'photons':
        d = np.diag(range(len(rho)))
        return np.real(np.trace(d @ rho))
    if a == 'energy':
        d = np.diag(range(len(rho)))
        return np.real(np.trace(d@rho))+1/2
    if a == 0:
        return np.log(np.sum(np.abs(la) != 0))
    if a == 1:
        return -np.sum(la * np.log(la))
    if a == 'inf' or np.isinf(a):
        return - np.log(np.max(np.abs(la)))
    return (1 / (1 - a)) * np.log(np.sum(la ** a))


def rho_eig(rho):
    eigval = np.real(np.linalg.eig(rho)[0])
    eigvec = np.transpose(np.linalg.eig(rho)[1])
    eigvec_sort = [vec for _, vec in sorted(zip(eigval, eigvec), key=operator.itemgetter(0))][::-1]
    eigval_sort = np.sort(eigval)[::-1]
    return np.array([eigval_sort, eigvec_sort], dtype=object)


def rho_eigval(rho):
    return rho_eig(rho)[0]


def rho_eigvec(rho):
    return rho_eig(rho)[1]


def vec_s(vec, a=1):
    # Compute the Von Neumann entropy (or Rényi entropy) of a vector
    vec = vec.astype(float)
    vec = np.delete(vec, np.where(vec == 0))
    if a == 0:
        return np.log(np.sum(np.abs(vec) != 0))
    if a == 1:
        return -np.sum(vec * np.log(np.abs(vec)))
    if a == 'inf' or np.isinf(a):
        return - np.log(np.max(np.abs(vec)))
    return (1 / (1 - a)) * np.log(np.sum(vec ** a))


def rho_phase(rho, phi):
    n = len(rho)
    nlist = np.arange(0, n)
    uphi = np.diag(np.exp(-1j*phi*nlist))
    return uphi @ rho @ np.conj(np.transpose(uphi))