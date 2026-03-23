import numpy as np
import scipy


def beam_splitter(i, k, n, eta=0.5):
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


def two_mode_squeezer(i, k, n, g=2):
    # Amplitude of transition in a TMS
    # tms_ikn = <n,m|Utms_lba|i,k> with n-m=i-kxs
    return beam_splitter(i, n + k - i, n, 1 / g) / np.sqrt(g)


def rescaling_map(m, n, q, r, s):
    """
    Transition of the rescaling map:
    Tr[L_s[|m><n|] |q><r|]

    Non-zero only if n - m == q - r.
    """
    if n < m:
        return rescaling_map(n, m, r, q, s)

    if n - m != q - r:
        return 0

    if s == 1:
        return 1 if (m == q and n == r) else 0
    

    prefactor = (
        2 / (s**2 + 1)
        * (-1)**m
        * np.sqrt(
            scipy.special.factorial(n)
            * scipy.special.factorial(r)
            / (scipy.special.factorial(m) * scipy.special.factorial(q))
        )
        * (2 * s / (s**2 + 1))**(n - m)
        * ((s**2 - 1) / (s**2 + 1))**(m + r)
    )

    summation = 0
    for k in range(min(m, r) + 1):
        term = (
            (-1)**k
            * scipy.special.comb(m, k, exact=True)
            * scipy.special.comb(q, r - k, exact=True)
            * (2 * s / (1 - s**2))**(2 * k)
        )
        summation += term

    return prefactor * summation