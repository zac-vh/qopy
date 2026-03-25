import numpy as np
import math
import qopy.state_space.transitions
from scipy.special import comb

def vertigo(rho, t, normalized=True):
    if t == 1:
        return rho
    N = len(rho)
    rout = np.zeros([N, N], dtype=complex)
    for m in range(N):
        for k in range(m+1):
            ammk = ((t - 1) ** m
                   * (math.factorial(m) / (math.factorial(k) * math.factorial(m - k)))
                   * (t / (t - 1)) ** k
                   )
            rout[k, k] = rout[k, k] + rho[m, m] * ammk
        for n in range(m+1, N):
            for k in range(m+1):
                amnk = (np.sqrt(t)**(n-m)
                      *(t-1)**m
                      *np.sqrt(math.factorial(m)/math.factorial(n))
                      *(math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))
                      *(t/(t-1))**(m-k)
                      *np.sqrt(math.factorial(n-k)/math.factorial(m-k))
                      )
                rout[m - k, n - k] = rout[m - k, n - k] + rho[m, n] * amnk
                rout[n - k, m - k] = rout[n - k, m - k] + rho[n, m] * amnk
    if normalized:
        rout = rout/np.sum(np.diag(rho)*(2*t-1)**(np.arange(N)))
    return rout


def pure_loss_channel(rho, eta):
    N = len(rho)
    if eta == 1: return rho
    if eta == 0:
        r0 = np.zeros_like(rho)
        r0[0, 0] = 1.0
        return r0
    rout = np.zeros_like(rho)
    n_indices = np.arange(N)
    eta_factors = np.sqrt(eta)**(n_indices[:, None] + n_indices[None, :])
    for k in range(N):
        m = n_indices[:N-k]
        c_k = np.sqrt(comb(m + k, k))
        coeffs_k = (1 - eta)**k * np.outer(c_k, c_k)
        rout[:N-k, :N-k] += coeffs_k * rho[k:, k:]   
    return rout * eta_factors


def quantum_amplifier_channel(rho, g, N_out=None):
    N_in = rho.shape[0]
    if N_out is None:
        N_out = N_in
    if g == 1:
        res = np.zeros((N_out, N_out), dtype=complex)
        m = min(N_in, N_out)
        res[:m, :m] = rho[:m, :m]
        return res
    rout = np.zeros((N_out, N_out), dtype=complex)
    n_in = np.arange(N_in)
    g_diag = g**(-n_in / 2.0)
    rho_scaled = rho * np.outer(g_diag, g_diag)
    for l in range(N_out):
        size = min(N_in, N_out - l)
        if size <= 0:
            break
        indices_i = np.arange(l, l + size)
        c_l = np.sqrt(comb(indices_i, l))
        coeffs_l = ((g - 1)**l / g**(l + 1)) * np.outer(c_l, c_l)
        rout[l:l+size, l:l+size] += coeffs_l * rho_scaled[:size, :size]
    return rout


def rescaling_map(rho, s, N_out):
    eta = 2*s**2/(s**2+1)
    gain = (s**2+1)/2
    rho_plc = pure_loss_channel(rho, eta)
    return quantum_amplifier_channel(rho_plc, gain, N_out)


def rescaling_map_old(rho, s, Nout=None):
    if Nout is None:
        Nout = len(rho)
    
    if s == 1 and Nout == len(rho):
        return rho.copy()
    
    N = len(rho)
    rout = np.zeros((Nout, Nout), dtype=complex)
    
    for m in range(N):
        for n in range(N):
            if rho[m, n] == 0:
                continue
            for q in range(Nout):
                r = q - (n - m)
                if r < 0 or r >= Nout:
                    continue
                amp = qopy.state_space.transitions.rescaling_map(m, n, q, r, s)
                rout[r, q] += amp * rho[m, n]
    
    return rout