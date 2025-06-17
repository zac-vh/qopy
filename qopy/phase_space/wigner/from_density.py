import numpy as np
import qopy.phase_space.wigner as wig
from qopy.phase_space.wavefunction import ket_to_psi, get_xl
from qopy.utils.linalg import matrix_eigenvalues_eigenvectors

def direct(rho, rl, nr, isherm=True):
    n = rho.shape[0]
    w = np.zeros((nr, nr), dtype=complex)
    for i in range(n):
        for j in range(i+1 if isherm else n):
            rij = rho[i, j]
            if rij == 0:
                continue
            wij = phase_space.cross_wigner.fock.grid(i, j, rl, nr)
            if isherm:
                if i != j:
                    w += 2*np.real(rij * wij)
                if i == j:
                    w += rij * wij
            else:
                w += rij * wij
    return np.real(w) if isherm else w


def cached(rho, wijset, isherm=True):
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


def psi(rho, rl, nr):
    # Compute the Wigner function from the wavefunction of the eigenvectors of rho
    # More efficient when N >> 1
    xl = get_xl(rl, nr)
    w = np.zeros((nr, nr), dtype=complex)
    eigvals, eigkets = matrix_eigenvalues_eigenvectors(rho)
    N = len(eigvals)
    for j in range(N):
        psij = ket_to_psi(eigkets[j], xl)
        w += eigvals[j] * wig.from_psi(psij, rl, nr)
    return np.real(w)