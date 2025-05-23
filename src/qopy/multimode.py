import numpy as np
import scipy

def wig2m_int(W, rl):
    # Integrate the 2-mode Wigner function
    nr = len(W)
    x = np.linspace(-rl / 2, rl / 2, nr)
    return scipy.integrate.simpson(scipy.integrate.simpson(scipy.integrate.simpson(scipy.integrate.simpson(W, x), x=x), x=x), x=x)


def wig2m_ptrace(W, rl, mode=2):
    nr = len(W)
    x = np.linspace(-rl/2, rl/2, nr)
    if mode==2:
        w = scipy.integrate.simpson(scipy.integrate.simpson(W, x=x), x=x)
    if mode==1:
        w = scipy.integrate.simpson(scipy.integrate.simpson(W, x=x, axis=0), x=x, axis=0)
    return w


def wig2m_h(W, rl):
    # Compute the Wigner entropy of a 2-mode Wigner function
    nr = len(W)
    x = np.linspace(-rl / 2, rl / 2, nr)
    Wlog = np.zeros([nr, nr, nr, nr])
    Wlog[np.nonzero(W)] = np.log(np.abs(W[np.nonzero(W)]))
    return - wig2m_int(W * Wlog, rl)



def trace_2m(rho):
    # Compute the trace of a 2m state
    # Convention is |i,j><k,l|
    ni = np.shape(rho)[0]
    nj = np.shape(rho)[1]
    tr = 0
    for i in range(ni):
        for j in range(nj):
            tr += rho[i][j][i][j]
    return tr


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
