import numpy as np
import math


def disp_mat(m, n, alpha):
    # return the value of <m|D(alpha)|n>
    dmn = 0
    for p in range(np.min([m, n])+1):
        dmn = dmn + np.sqrt(float(math.factorial(m)*math.factorial(n)))*(-1)**(n-p)/(math.factorial(p)*math.factorial(m-p)*math.factorial(n-p))*alpha**(m-p)*np.conj(alpha)**(n-p)
    dmn = np.exp(-(1/2)*np.abs(alpha)**2)*dmn
    return dmn


def rho_rotate(rho, phi):
    N = len(rho)
    rotmat = np.diag(np.exp(-1j*phi*np.arange(N)))
    rhorot = rotmat @ rho @ np.conj(rotmat)
    return rhorot


def rho_mean(rho):
    n = len(rho)
    a = 0+0j
    for i in range(n-1):
        a = a + rho[i, i+1]*np.sqrt(i+1)
    xm = np.sqrt(2)*np.real(a)
    pm = -np.sqrt(2)*np.imag(a)
    return np.array([xm, pm])


def rho_covmat(rho, mean=None):
    n = len(rho)
    if mean is None:
        [xm, pm] = rho_mean(rho)
    elif not (isinstance(mean, list) or (isinstance(mean, np.ndarray)) or (isinstance(mean, tuple))):
        xm = np.sqrt(2)*np.real(mean)
        pm = np.sqrt(2)*np.imag(mean)
    else:
        xm = mean[0]
        pm = mean[1]

    nm = rho_s(rho, 'photons')
    aa = 0+0j
    for i in range(n-2):
        aa = aa + rho[i, i+2]*np.sqrt((i+1)*(i+2))
    xxm = np.real(aa)+nm+1/2-xm**2
    ppm = -np.real(aa)+nm+1/2-pm**2
    xpm = -np.imag(aa)-xm*pm
    cov = np.array([[xxm, xpm], [xpm, ppm]])
    return cov