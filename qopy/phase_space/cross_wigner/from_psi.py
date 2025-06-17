import numpy as np


def from_psi(f, g, rl, nr):
    """
    Fast version: vectorized over p, avoids double for-loop and simpson.
    """
    x = np.linspace(-rl / 2, rl / 2, nr)
    dx = x[1] - x[0]
    wij = np.zeros((nr, nr), dtype=complex)

    phase = np.exp(2j * np.outer(x, x))  # shape (nr, nr)

    for xk in range(nr):
        fk = np.conj(f[xk:xk + nr])       # shape (nr,)
        gk = np.flip(g[xk:xk + nr])       # shape (nr,)
        integrand = fk * gk               # shape (nr,)
        wij[xk, :] = dx * np.dot(phase, integrand) / np.pi  # dot over axis=1

    return wij