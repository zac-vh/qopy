'''
Contains all the function that modify phase-space distributions
F(W) -> W

'''

import numpy as np
import math
import scipy

#from qopy.phase_space import wig_int

def wig_trunc(w, rl):
    # Makes W a non-negative distribution by averaging the negative parts of W with the lowest positive parts
    nr = len(w)
    dxdp = (rl / (nr - 1)) ** 2
    vneg = -wig_int(w * (w < 0), rl)
    wpos = w * (w > 0)
    min_val = np.sort(np.reshape(wpos, nr ** 2))
    min_val = np.unique(min_val[min_val != 0])
    for i in range(len(min_val)):
        min_ind = np.where(wpos == min_val[i])
        shape = np.shape(min_ind)
        for j in range(shape[1]):
            jx = min_ind[0][j]
            jp = min_ind[1][j]
            if vneg >= w[jx][jp] * dxdp:
                vneg = vneg - wpos[jx][jp] * dxdp
                wpos[jx][jp] = 0
            else:
                vneg = 0
                wpos[jx][jp] = wpos[jx][jp] - vneg / dxdp
                break
        if vneg == 0:
            break
    return wpos


def get_rl_fft(nr):
    return np.sqrt(2 * math.pi / nr) * (nr - 1)


def wig_fft(w, rl, rescale=True):
    # The rescaling factor is equal to 2pi/(dx**2*nr)=2pi*(nr-1)^2/(n*r^2), which scales like n/r^2.
    # rl = np.sqrt(2*math.pi/nr)*(nr-1)
    nr = len(w)
    x = np.arange(0, nr)
    mx, mp = np.meshgrid(x, x)
    dx = rl/(nr-1)
    phase = np.exp(1j*math.pi*rl*(mx+mp)/(dx*nr))
    wf = phase*scipy.fft.fft2(w*phase)
    wf = np.exp(-1j*math.pi*rl**2/(nr*dx**2))*wf
    wf = 2*math.pi/(dx**2*nr**2)*wf
    factor = 2 * math.pi / (dx ** 2 * nr)
    if rescale and factor != 1:
        wf = wig_rescale(wf, factor)
        print('FFT rescaling: '+str(factor))
    return wf


def wig_ifft(w, rl, rescale=True):
    # The rescaling factor is equal to 2pi/(dx**2*nr)=2pi*(nr-1)^2/(n*r^2), which scales like n/r^2.
    # rl = np.sqrt(2*math.pi/nr)*(nr-1)
    nr = len(w)
    x = np.arange(0, nr)
    mx, mp = np.meshgrid(x, x)
    dx = rl/(nr-1)
    phase = np.exp(-1j*math.pi*rl*(mx+mp)/(dx*nr))
    wf = phase*scipy.fft.ifft2(w*phase)*nr**2
    wf = np.exp(1j*math.pi*rl**2/(nr*dx**2))*wf
    wf = 2*math.pi/(dx**2*nr**2)*wf
    factor = 2 * math.pi / (dx ** 2 * nr)
    if rescale and factor != 1:
        wf = wig_rescale(wf, factor)
        print('IFFT rescaling: '+str(factor))
    return wf


def wig_conv(w1, w2, rl, mode='fft', shift=True):
    # Convolve w1 and w2 using FFT
    nr = len(w1)
    if mode == 'fft':
        w = scipy.signal.fftconvolve(w1, w2, mode='same') * (rl / (nr-1)) ** 2
        if np.max(np.abs(np.imag(w1))) == 0 and np.max(np.abs(np.imag(w2))) == 0:
            w = np.real(w)
    elif mode == 'pad':
        w = scipy.signal.convolve2d(w1, w2, mode='same') * (rl / (nr-1)) ** 2
    if shift:
        d = rl/(nr-1)
        w = wig_disp(w, [-d/2, -d/2], rl)
    return w


def wig_rescale(w, s, k=5):
    # Rescale the Wigner function from the origin with a factor c
    # k is the interpolation parameter (default was 3)
    nr = len(w)
    x = np.linspace(-(nr - 1) / 2, (nr - 1) / 2, nr)
    w_interp_real = scipy.interpolate.RectBivariateSpline(x, x, np.real(w), kx=k, ky=k)
    wc = w_interp_real(x / s, x / s)
    if np.iscomplexobj(w):
        w_interp_imag = scipy.interpolate.RectBivariateSpline(x, x, np.imag(w), kx=k, ky=k)
        wc = wc + w_interp_imag(x / s, x / s) * 1j
    return wc/s**2

def wig_squeeze(w, s, k=5):
    nr = len(w)
    x = np.linspace(-(nr - 1) / 2, (nr - 1) / 2, nr)
    w_interp_real = scipy.interpolate.RectBivariateSpline(x, x, np.real(w), kx=k, ky=k)
    ws = w_interp_real(s * x, x / s)
    if np.iscomplexobj(w):
        w_interp_imag = scipy.interpolate.RectBivariateSpline(x, x, np.imag(w), kx=k, ky=k)
        ws = ws + w_interp_imag(s * x, x / s) * 1j
    return ws

def wig_rotate(w, phi, k=5, pref=True):
    phi = 180*phi/math.pi
    wr_real = scipy.ndimage.rotate(np.real(w), phi, reshape=False, order=k, mode='constant', cval=0.0, prefilter=pref)
    wr = wr_real
    if np.iscomplexobj(w):
        wr_imag = scipy.ndimage.rotate(np.imag(w), phi, reshape=False, order=k, mode='constant', cval=0.0, prefilter=pref)
        wr = wr + wr_imag * 1j
    return wr


def wig_disp(w, d, rl, k=5):
    nr = len(w)
    if not (isinstance(d, list) or (isinstance(d, np.ndarray))):
        d = [np.real(d), np.imag(d)]
    x = np.linspace(-rl / 2, rl / 2, nr)
    w_interp_real = scipy.interpolate.RectBivariateSpline(x, x, np.real(w), kx=k, ky=k)
    ws = w_interp_real(x - d[0], x - d[1])
    if np.iscomplexobj(w):
        w_interp_imag = scipy.interpolate.RectBivariateSpline(x, x, np.imag(w), kx=k, ky=k)
        ws = ws + w_interp_imag(x - d[0], x - d[1]) * 1j
    return ws


def wig_sympl(w, rl):
    # Applies the symplectic unitary that minimizes the energy of w
    w = wig_disp(w, -wig_mean(w, rl), rl)
    cov = wig_covmat(w, rl)
    eigvals, eigvecs = np.linalg.eig(cov)
    theta = np.arccos(eigvecs[0][0])
    if eigvecs[1][0] < 0:
        theta = math.pi - theta
    w = wig_rotate(w, -theta)
    cov = wig_covmat(w, rl)
    sq = (cov[0, 0]/cov[1, 1])**(1/4)
    w = wig_squeeze(w, sq)
    return w


def wig_gradient(w, rl):
    nr = len(w)
    gw = np.gradient(w, rl / (nr - 1), edge_order=2)
    #gwx = gw[0]
    #gwp = gw[1]
    return gw


def wig_laplace(w, rl, power=1):
    if power == 0:
        return w
    nr = len(w)
    gw = wig_gradient(w, rl)
    gwx = gw[0]
    gwxx = wig_gradient(gwx, rl)[0]
    gwp = gw[1]
    gwpp = wig_gradient(gwp, rl)[1]
    wout = gwxx + gwpp
    return wig_laplace(wout, rl, power-1)