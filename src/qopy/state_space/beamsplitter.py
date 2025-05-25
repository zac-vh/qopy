import numpy as np
import scipy
from qopy.state_space.density import rho_trim


def bs_transition_amplitude(i, k, n, eta=0.5):
    #Return the transition amplitude of the beam-splitter: <i,k|U|n,i+k-n>
    if (i < 0) or (k < 0) or (n < 0) or (i+k < n):
        return 0
    if eta == 0:
        if n == k:
            return (-1)**k
        else:
            return 0
    if eta == 1:
        if i == n:
            return 1
        else:
            return 0
    if k >= n:
        res = scipy.special.comb(k, n, exact=True)*scipy.special.hyp2f1(-i, -n, 1+k-n, eta/(eta-1))
    else:
        res = scipy.special.comb(i, n-k, exact=True)*scipy.special.hyp2f1(-k, -i-k+n, 1-k+n, eta/(eta-1))*(eta/(eta-1))**(n-k)
    return (-1)**i*np.sqrt((1-eta)**(i+n)*eta**(k-n)*math.factorial(i+k-n)*math.factorial(n)/(math.factorial(i)*math.factorial(k)))*res


def tms_transition_amplitude(i, k, n, g=2):
    # Amplitude of transition in a TMS
    # tms_ikn = <n,m|Utms_lba|i,k> with n-m=i-kxs
    b = bs_transition_amplitude(i, n + k - i, n, 1 / g) / np.sqrt(g)
    return b


def bs_output_1m(rho1, rho2, eta=0.5, mode=1):
    if mode == 1:
        return bs_output_1m_mode1(rho1, rho2, eta)
    else:
        return bs_output_1m_mode2(rho1, rho2, eta)


def bs_output_1m_mode1(rho1, rho2, eta=0.5):
    rho1 = rho_trim(rho1)
    rho2 = rho_trim(rho2)
    n1 = len(rho1)
    n2 = len(rho2)
    n = n1+n2-1
    rho = np.zeros([n, n], dtype=complex)
    for m in range(n):
        for r in range(n):
            rmr = 0
            for i in range(n1):
                for j in range(n2):
                    for k in range(n1):
                        l = i+j-m+r-k
                        if 0 <= l <= (n2-1):
                            rmr = rmr + rho1[i][k] * rho2[j][l] * bs_transition_amplitude(i, j, m, eta) * bs_transition_amplitude(k, l, r, eta)
            rho[m][r] = rmr
    return rho


def bs_output_1m_mode2(rho1, rho2, eta=0.5):
    rho1 = rho_trim(rho1)
    rho2 = rho_trim(rho2)
    n1 = len(rho1)
    n2 = len(rho2)
    ntot = n1+n2-1
    rho = np.zeros([ntot, ntot], dtype=complex)
    for n in range(ntot):
        for s in range(ntot):
            rns = 0
            for i in range(n1):
                for j in range(n2):
                    for k in range(n1):
                        l = i+j-n+s-k
                        if 0 <= l <= (n2-1):
                            t = i+j-n
                            rns = rns + rho1[i][k] * rho2[j][l] * bs_transition_amplitude(i, j, t, eta) * bs_transition_amplitude(k, l, t, eta)
            rho[n][s] = rns
    return rho


def bs_output_2m_ket(ket_p, ket_q, eta=0.5):
    # Compute the 2m output of a BS fed by the kets |p> and |q>
    ket_p = np.trim_zeros(ket_p, 'b')
    ket_q = np.trim_zeros(ket_q, 'b')
    n_phot_p = len(ket_p) - 1
    n_phot_q = len(ket_q) - 1
    n_phot_max = n_phot_p + n_phot_q
    rho = np.zeros([n_phot_max + 1, n_phot_max + 1, n_phot_max + 1, n_phot_max + 1], dtype=complex)
    # rho = sum_nmrs rho_nmrs |n,m><r,s|
    for n in range(n_phot_max + 1):
        for m in range(n_phot_max + 1):
            for r in range(n_phot_max + 1):
                for s in range(n_phot_max + 1):
                    rho_nmrs = 0
                    for i in range(n_phot_p + 1):
                        for k in range(n_phot_p + 1):
                            j = n + m - i
                            l = r + s - k
                            if (0 <= j <= n_phot_q) and (0 <= l <= n_phot_q):
                                rho_nmrs += ket_p[i] * ket_q[j] * np.conj(ket_p[k]) * np.conj(ket_q[l]) * \
                                            bs_transition_amplitude(i, j, n, eta) * bs_transition_amplitude(k, l, r, eta)
                    rho[n][m][r][s] = rho_nmrs
    return rho


def bs_unitary_2m(rho2m, eta=0.5):
    n_phot_mode_1 = np.shape(rho2m)[0] - 1
    n_phot_mode_2 = np.shape(rho2m)[1] - 1
    n_phot_max = n_phot_mode_1 + n_phot_mode_2
    rho_out = np.zeros([n_phot_max + 1, n_phot_max + 1, n_phot_max + 1, n_phot_max + 1], dtype=complex)
    for a in range(n_phot_max + 1):
        for b in range(n_phot_max + 1):
            for c in range(n_phot_max + 1):
                for d in range(n_phot_max + 1):
                    rho_abcd = 0
                    for r in range(n_phot_mode_1 + 1):
                        for u in range(n_phot_mode_1 + 1):
                            s = a + b - r
                            v = c + d - u
                            if (0 <= s <= n_phot_mode_2) and (0 <= v <= n_phot_mode_2):
                                rho_abcd += rho2m[r][s][u][v] * bs_transition_amplitude(r, s, a, eta) * bs_transition_amplitude(u, v, c, eta)
                    rho_out[a][b][c][d] = rho_abcd
    return rho_out


def sigma_mn(m, n, eta=0.5):
    out = np.zeros(m+n+1)
    for i in range(m+n+1):
        out[i] = bs_transition_amplitude(m, n, i, eta) ** 2
    return out
