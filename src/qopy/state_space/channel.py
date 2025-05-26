import numpy as np
import math

def vertigo(rho, s, normalized=False):
    if s == 1:
        return rho
    N = len(rho)
    rout = np.zeros([N, N], dtype=complex)
    for m in range(N):
        for k in range(m+1):
            ammk = ((s - 1) ** m
                   * (math.factorial(m) / (math.factorial(k) * math.factorial(m - k)))
                   * (s / (s - 1)) ** k
                   )
            rout[k, k] = rout[k, k] + rho[m, m] * ammk
        for n in range(m+1, N):
            for k in range(m+1):
                amnk = (np.sqrt(s)**(n-m)
                      *(s-1)**m
                      *np.sqrt(math.factorial(m)/math.factorial(n))
                      *(math.factorial(n)/(math.factorial(k)*math.factorial(n-k)))
                      *(s/(s-1))**(m-k)
                      *np.sqrt(math.factorial(n-k)/math.factorial(m-k))
                      )
                rout[m - k, n - k] = rout[m - k, n - k] + rho[m, n] * amnk
                rout[n - k, m - k] = rout[n - k, m - k] + rho[n, m] * amnk
    if normalized:
        rout = rout/np.sum(np.diag(rho)*(2*s-1)**(np.arange(N)))
    return rout


def pure_loss_channel(rho, eta):
    N = len(rho)
    if eta == 1:
        return rho
    if eta == 0:
        r0 = np.zeros([N, N], dtype=complex)
        r0[0, 0] = 1
        return r0
    rout = np.zeros([N, N], dtype=complex)
    A = np.diag(np.sqrt(np.arange(1, N)), 1)
    Adag = np.diag(np.sqrt(np.arange(1, N)), -1)
    Neta = np.diag(np.sqrt(eta)**np.arange(N))
    Ak = np.eye(N)
    Adagk = np.eye(N)
    for k in range(N):
        Dk = Neta @ Ak @ rho @ Adagk @ Neta
        rout += ((1 - eta) ** k / math.factorial(k)) * Dk
        Ak = Ak @ A
        Adagk = Adagk @ Adag
    return rout
