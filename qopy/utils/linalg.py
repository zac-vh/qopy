import numpy as np
import scipy
import operator

def square_trim(M, tol=0):
    M = np.asarray(M)
    while M.shape[0] > 1:
        last = M.shape[0] - 1
        row = np.abs(M[last, :])
        col = np.abs(M[:, last])
        if np.all(row <= tol) and np.all(col <= tol):
            M = M[:last, :last]
        else:
            break
    return M


def convolve_matrix(M1, M2):
    Mout = scipy.signal.convolve2d(M1, M2, mode='full')
    return Mout


def matrix_eigenvalues_eigenvectors(M):
    # Assume M is Hermitian
    eigval = np.real(np.linalg.eig(M)[0])
    eigvec = np.transpose(np.linalg.eig(M)[1])
    eigvec_sort = [vec for _, vec in sorted(zip(eigval, eigvec), key=operator.itemgetter(0))][::-1]
    eigval_sort = np.sort(eigval)[::-1]
    return np.array([eigval_sort, eigvec_sort], dtype=object)


def matrix_eigenvalues(rho):
    return matrix_eigenvalues_eigenvectors(rho)[0]


def matrix_eigenvectors(rho):
    return matrix_eigenvalues_eigenvectors(rho)[1]


def renyi_entropy_vector(vec, alpha):
    # Compute the Rényi entropy of a probability vector
    vec = vec.astype(float)
    vec = np.delete(vec, np.where(vec == 0))
    if alpha == 0:
        return np.log(np.sum(np.abs(vec) != 0))
    if alpha == 1:
        return -np.sum(vec * np.log(np.abs(vec)))
    if np.isinf(alpha):
        return - np.log(np.max(np.abs(vec)))
    return (1 / (1 - alpha)) * np.log(np.sum(vec ** alpha))


def renyi_entropy_matrix(rho, alpha):
    la = matrix_eigenvalues(rho)
    return renyi_entropy_vector(la, alpha)