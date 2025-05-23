'''
Contains all the function that extract infromation from a phasespace distributions
F(W) -> real
'''


import numpy as np
import scipy
import math
import skimage


def wig_mean(w, rl):
    nr = len(w)
    x = np.linspace(-rl / 2, rl / 2, nr)
    mx, my = np.meshgrid(x, x, indexing='ij')
    xm = wig_int(w*mx, rl)
    ym = wig_int(w*my, rl)
    return np.array([xm, ym])


def wig_covmat(w, rl, mean=None):
    nr = len(w)
    x = np.linspace(-rl / 2, rl / 2, nr)
    mx, my = np.meshgrid(x, x, indexing='ij')
    if mean is None:
        [dx, dy] = wig_mean(w, rl)
    elif not (isinstance(mean, list) or (isinstance(mean, np.ndarray)) or (isinstance(mean, tuple))):
        dx = np.sqrt(2)*np.real(mean)
        dy = np.sqrt(2)*np.imag(mean)
    else:
        dx = mean[0]
        dy = mean[1]
    vxx = wig_int(mx**2*w, rl)-dx**2
    vyy = wig_int(my**2*w, rl)-dy**2
    vxy = wig_int(mx*my*w, rl)-dx*dy
    v = np.array([[vxx, vxy], [vxy, vyy]])
    return v


def wig_marg(w, rl, theta=0):
    # Gives the marginal probability
    if theta == 'x':
        theta = 0
    if theta == 'p':
        theta = np.pi/2
    nr = len(w)
    x = np.linspace(-rl / 2, rl / 2, nr)
    w = wig_rotate(w, -theta)
    return scipy.integrate.simpson(w, x=x)


def wig_int(w, rl):
    # Integrate the Wigner function
    nr = len(w)
    x = np.linspace(-rl / 2, rl / 2, nr)
    return scipy.integrate.simpson(scipy.integrate.simpson(w, x=x), x=x)


def wig_h(w, rl):
    # Compute the Wigner(Rényi) entropy of the Wigner function
    nr = len(w)
    x = np.linspace(-rl / 2, rl / 2, nr)
    wlog = np.zeros([nr, nr])
    wlog[np.nonzero(w)] = np.log(np.abs(w[np.nonzero(w)]))
    return - wig_int(w * wlog, rl)


def wig_h_trunc(w, rl):
    nr = len(w)
    dxp = (rl/(nr-1))**2
    wd = np.sort(np.reshape(w, nr ** 2))[::-1]
    sum = 0
    imax = nr**2
    for i in range(nr**2):
        sum = sum + wd[i]*dxp
        if sum >1:
            imax = i
            break
    wt = wd[:imax]
    logwt = np.nan_to_num(np.log(wt))
    return -np.sum(wt*logwt)*dxp


def wig_pnorm(w, rl, p):
    if p == 0:
        wp = (np.abs(w) != 0)
        pnorm = wig_int(wp, rl)
        return pnorm
    if math.isinf(p):
        pnorm = np.max(np.abs(w))
        return pnorm
    wp = np.abs(w)**p
    pnorm = wig_int(wp, rl)**(1/p)
    return pnorm


def wig_renyi(w, rl, alpha):
    if alpha == 0:
        wp = (np.abs(w) != 0)
        hr = np.log(wig_int(wp, rl))
        return hr
    if math.isinf(alpha):
        hr = -np.log(np.max(np.abs(w)))
        return hr
    if alpha == 1:
        return wig_h(w, rl)
    wa = np.abs(w)**alpha
    hr = (1/(1-alpha))*np.log(wig_int(wa, rl))
    return hr


def wig_fishmat(w, rl):
    nr = len(w)
    gw = np.gradient(np.abs(w), rl/(nr-1))
    gwx = gw[0]
    gwp = gw[1]
    invw = np.zeros([nr, nr])
    invw[np.nonzero(w)] = 1/w[np.nonzero(w)]
    jxx = wig_int(gwx**2*invw, rl)
    jpp = wig_int(gwp**2*invw, rl)
    jxp = wig_int(gwx*gwp*invw, rl)
    j = np.array([[jxx, jxp], [jxp, jpp]])
    return j


def wig_radon(w, rl, ntheta=None):
    nr = len(w)
    if ntheta is None:
        ntheta = nr
    x = np.arange(nr)-(nr // 2)
    mx, my = np.meshgrid(x, x)
    w[mx**2+my**2 > (nr**2/4)] = 0
    wr = skimage.transform.radon(w, np.linspace(0, 360, ntheta, endpoint=False), circle=True)
    wr = wr*(rl/(nr-1))
    return wr


def wig_to_rho(wijset, w, rl):
    # Build the density matrix rho from a Wigner function (in wij_set basis)
    n = np.shape(wijset)[0]
    rho = np.zeros([n, n], dtype=complex)
    for i in range(n):
        for j in range(n):
            wji = wijset[j][i]
            rho[i][j] = 2*math.pi * wig_int(w * wji, rl)
    return rho