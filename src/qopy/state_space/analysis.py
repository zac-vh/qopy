'''
Contains all the functions extracting information from (density) operators

'''

import numpy as np

def rho_mean(rho):
    n = len(rho)
    a = 0+0j
    for i in range(n-1):
        a = a + rho[i, i+1]*np.sqrt(i+1)
    xm = np.sqrt(2)*np.real(a)
    pm = -np.sqrt(2)*np.imag(a)
    return np.array([xm, pm])