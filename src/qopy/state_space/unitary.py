
def disp_mat(m, n, alpha):
    # return the value of <m|D(alpha)|n>
    dmn = 0
    for p in range(np.min([m, n])+1):
        dmn = dmn + np.sqrt(float(math.factorial(m)*math.factorial(n)))*(-1)**(n-p)/(math.factorial(p)*math.factorial(m-p)*math.factorial(n-p))*alpha**(m-p)*np.conj(alpha)**(n-p)
    dmn = np.exp(-(1/2)*np.abs(alpha)**2)*dmn
    return dmn


def bs_ikn(i, k, n, eta=0.5):
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


def bs_ikn_prec(i, k, n, eta=0.5, prec=50):
    mp.dps = prec
    eta = mp.mpf(eta)
    tot = mp.mpf(0)
    for p in range(max(0, n - k), min(i, n) + 1):
        term = (eta / (eta - 1)) ** p * mp.binomial(i, p) * mp.binomial(k, p + k - n)
        tot += term
    prefact = (-1) ** i * mp.sqrt(mp.factorial(n) * mp.factorial(i + k - n) / (mp.factorial(i) * mp.factorial(k))) * mp.sqrt((1 - eta) ** (n + i) / (eta ** (n - k)))
    return prefact*tot



def tms_ikn(i, k, n, g=2, prec=None):
    # Amplitude of transition in a TMS
    # tms_ikn = <n,m|Utms_lba|i,k> with n-m=i-kxs
    b = bs_ikn(i, n + k - i, n, 1/g, prec)/np.sqrt(g)
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
                            rmr = rmr + rho1[i][k]*rho2[j][l]*bs_ikn(i, j, m, eta)*bs_ikn(k, l, r, eta)
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
                            rns = rns + rho1[i][k]*rho2[j][l]*bs_ikn(i, j, t, eta)*bs_ikn(k, l, t, eta)
            rho[n][s] = rns
    return rho


def bs_output_1m_mode1_prec(rho1, rho2, eta, prec):
    rho1 = rho_trim(rho1)
    rho2 = rho_trim(rho2)
    n1 = len(rho1)
    n2 = len(rho2)
    n = n1+n2-1
    rho = np.zeros([n, n], dtype=complex)
    for m in range(n):
        for r in range(n):
            rmrRe = decimal.Decimal(0)
            rmrIm = decimal.Decimal(0)
            for i in range(n1):
                for j in range(n2):
                    for k in range(n1):
                        l = i+j-m+r-k
                        if 0 <= l <= (n2-1):
                            rmrbs = bs_ikn(i, j, m, eta, prec)*bs_ikn(k, l, r, eta, prec)
                            rmrRe = rmrRe + decimal.Decimal(np.real(rho1[i][k]*rho2[j][l]))*rmrbs
                            rmrIm = rmrIm + decimal.Decimal(np.imag(rho1[i][k]*rho2[j][l]))*rmrbs
            rho[m][r] = float(rmrRe)+float(rmrIm)*1j
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
                                            bs_ikn(i, j, n, eta) * bs_ikn(k, l, r, eta)
                    rho[n][m][r][s] = rho_nmrs
    return rho


def bs_unit_2m(rho2m, eta=0.5):
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
                                rho_abcd += rho2m[r][s][u][v] * bs_ikn(r, s, a, eta) * bs_ikn(u, v, c, eta)
                    rho_out[a][b][c][d] = rho_abcd
    return rho_out


def bs_output_mix(mix_p, mix_q, eta=0.5):
    # Compute the output of a BS fed by two mixtures of Fock states
    mix_p = np.trim_zeros(mix_p, 'b')
    mix_q = np.trim_zeros(mix_q, 'b')
    n_phot_p = len(mix_p) - 1
    n_phot_q = len(mix_q) - 1
    n_phot_max = n_phot_p + n_phot_q
    mix_out = np.zeros(n_phot_max + 1)
    for i in range(n_phot_p + 1):
        for j in range(n_phot_q + 1):
            for k in range(i + j + 1):
                mix_out[k] += mix_p[i] * mix_q[j] * bs_ikn(i, j, k, eta) ** 2
    return mix_out


def sigma_mn(m, n, eta=0.5, prec=None):
    out = np.zeros(m+n+1)
    if prec is None:
        for i in range(m+n+1):
            out[i] = bs_ikn(m, n, i, eta)**2
    else:
        for i in range(m+n+1):
            out[i] = float((bs_ikn_prec(m, n, i, eta, prec)**2).real)
    return out
