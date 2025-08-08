import numpy as np
import scipy
from qopy.state_space.density.base import trim


def transition_amplitude(i, k, n, eta=0.5):
    """
    Return the transition amplitude of the beam splitter: ⟨i, k| U_BS |n, i+k−n⟩
    where eta is the transmittance (0 ≤ eta ≤ 1).
    """
    if i < 0 or k < 0 or n < 0 or i + k < n:
        return 0
    if eta == 0:
        return (-1)**k if n == k else 0
    if eta == 1:
        return 1 if n == i else 0
    t = eta
    r = 1 - eta
    factor = (-1)**i * np.sqrt(
        r**(i + n) * t**(k - n) *
        scipy.special.factorial(i + k - n) *
        scipy.special.factorial(n) /
        (scipy.special.factorial(i) * scipy.special.factorial(k))
    )

    if k >= n:
        binom = scipy.special.comb(k, n, exact=True)
        hyper = scipy.special.hyp2f1(-i, -n, 1 + k - n, t / (t - 1))
        res = binom * hyper
    else:
        binom = scipy.special.comb(i, n - k, exact=True)
        z = t / (t - 1)
        hyper = scipy.special.hyp2f1(-k, -i - k + n, 1 - k + n, z)
        res = binom * hyper * z**(n - k)

    return factor * res


def bs_output_1m(rho1, rho2, eta=0.5, mode=1):
    if mode == 1:
        return output_1m_mode1(rho1, rho2, eta)
    else:
        return output_1m_mode2(rho1, rho2, eta)


def output_1m_mode1(rho1, rho2, eta=0.5):
    rho1 = trim(rho1)
    rho2 = trim(rho2)
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
                            rmr = rmr + rho1[i][k] * rho2[j][l] * transition_amplitude(i, j, m, eta) * transition_amplitude(k, l, r, eta)
            rho[m][r] = rmr
    return rho


def output_1m_mode2(rho1, rho2, eta=0.5):
    rho1 = trim(rho1)
    rho2 = trim(rho2)
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
                            rns = rns + rho1[i][k] * rho2[j][l] * transition_amplitude(i, j, t, eta) * transition_amplitude(k, l, t, eta)
            rho[n][s] = rns
    return rho


def output_2m_ket(ket_p, ket_q, eta=0.5):
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
                                            transition_amplitude(i, j, n, eta) * transition_amplitude(k, l, r, eta)
                    rho[n][m][r][s] = rho_nmrs
    return rho


def unitary_2m(rho2m, eta=0.5):
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
                                rho_abcd += rho2m[r][s][u][v] * transition_amplitude(r, s, a, eta) * transition_amplitude(u, v, c, eta)
                    rho_out[a][b][c][d] = rho_abcd
    return rho_out


def sigma_mn(m, n, eta=0.5):
    out = np.zeros(m+n+1)
    for i in range(m+n+1):
        out[i] = transition_amplitude(m, n, i, eta) ** 2
    return out
