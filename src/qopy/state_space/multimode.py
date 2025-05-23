import numpy as np

def tensor_to_matrix(T):
    #convert a NxNxNxN tensor into a N^2xN^2 matrix
    #convert a (ndim x ndim)^(nmode) tensor into ndim^nmode x ndim^nmode matrix
    nmode = round(len(T.shape)/2)
    ndim = T.shape[0]
    M = np.reshape(T, [ndim**nmode, ndim**nmode], order='F')
    return M


def matrix_to_tensor(M, nmode=2):
    #convert a N^2xN^2 matrix into a NxNxNxN tensor
    ndim = round(np.power(M.shape[0], 1/nmode))
    T = np.reshape(M, [ndim, ndim]*nmode, order='F')
    return T


def tensor_to_matrix_singlemode(T):
    #convert a NxNxNxN tensor into a NxN^3 matrix
    n = T.shape[0]
    T = T.transpose(1, 0, 2, 3)
    M = np.reshape(T, [n, n ** 3], order='F')
    return M


def density_tensor(r1, r2):
    # use the convention to use order |0,1><2,3|=|0><2|o|1><3|
    # to trace over second mode, trace over indices 1,3
    # to trace over first mode, trace over indices 0,2
    return np.transpose(np.tensordot(r1, r2, axes=0), axes=[0, 2, 1, 3])