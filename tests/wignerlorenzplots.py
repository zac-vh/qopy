import numpy as np, scipy as sp
from scipy import linalg
from scipy.linalg import logm, expm, eigh
from scipy.special import laguerre
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from math import factorial

import pickle

from qutip import Qobj, tracedist
from matplotlib.widgets import Cursor


import os
di=os.curdir

from functools import partial

##functions to return various rearrangements and lorenz curves
def inc(f):
    return np.sort(f,axis=None)

def dec(f):
    return np.sort(f,axis=None)[::-1]

def inclor(f):
    return np.append(np.array([0],dtype=np.float16),np.cumsum(inc(f)))

def declor(f):
    return np.append(np.array([0],dtype=np.float16),np.cumsum(dec(f)))

def inclorneg(f):
    return np.append(np.array([0],dtype=np.float16),np.cumsum(inc(np.where(f<=0,f,0))))

def declorpos(f):
    return np.append(np.array([0],dtype=np.float16),np.cumsum(dec(np.where(f>=0,f,0))))

def inclorneg_sparse(f,l):
    raw=np.append(np.array([0],dtype=np.float16),np.cumsum(inc(np.where(f<=0,f,0))))
    pad=raw[-1]
    return np.append(raw,pad)

def declorpos_sparse(f,l):
    raw=np.append(np.array([0],dtype=np.float16),np.cumsum(dec(np.where(f>=0,f,0))))
    pad=raw[-1]
    return np.append(raw,pad)


##define Wigner functions of various states
def fock(n,x,p):
    coef=np.zeros(n+1)
    coef[n]=1
    return  (-1)**n * 2/np.pi * np.exp(-2*(x**2+p**2))*  np.polynomial.laguerre.lagval(4*(x**2+p**2), coef)   

def thermal(nph,x,p):
    return  2/np.pi/(1+2*nph) * np.exp(-2*(x**2+p**2)/(1+2*nph))

def coherent(alpha,x,p):
    return 2/np.pi * np.exp(-2*((x-np.real(alpha))**2+(p-np.imag(alpha))**2))

def squeezed(alpha,r,x,p):
    return 2/np.pi * np.exp(-2*((x-np.real(alpha))**2*np.exp(-2*r)+(p-np.imag(alpha))**2*np.exp(2*r)))

def squeezedcubicphase(g,P,s,x,p): #g=gamma  
    return 4/np.sqrt(8*np.pi**3*np.exp(2*s))*np.exp(-2*x**2/(np.exp(2*s))) * integrate.quad_vec(lambda yy: (2*np.cos(12*g*yy**3+2*yy*(3*g*(x**2)-p+P)) * np.exp(-yy**2/(2*np.exp(2*s)))),0,np.inf)[0]

def ONstate(a,n,x,p):
    return 1/(1+np.abs(a)**2)*fock(0,x,p)+np.abs(a)**2/(1+np.abs(a)**2)*fock(n,x,p)+1/(1+np.abs(a)**2)/np.sqrt(factorial(n))/(2*np.pi)*np.exp(-p**2-x**2)*(a*(x-1j*p)**n+np.conj(a)*(x+1j*p)**n)
    

##generate lattice for sampling
lim=10
npts=201
x = np.linspace(-lim, lim, npts)
y = np.linspace(-lim, lim, npts)
dx=2*lim/(npts-1)
dy=dx
dsq=dx*dy
# full coordinate arrays
xx, yy = np.meshgrid(x, y)


def lorplotroutine(farr): #input is an array of functions 
    allzz=np.array([1])
    for i,f in enumerate(farr):
        temp=f(xx,yy)*dsq
        temp=temp[np.abs(temp)>1e-10] #discard elements close to zero, manual sparse
        allzz=np.kron(allzz,temp)
    l=npts**(2*np.size(farr))+1    
    dl=declorpos_sparse(allzz,l)
    il=inclorneg_sparse(allzz,l)
    xarr=dsq**np.size(farr)*np.append(np.linspace(0,np.size(dl)-2,np.size(dl)-1),np.array(l-1))
    return (xarr,dl,il,allzz)


## example
n=2 #num. input modes
k=2 #num output modes
inputarr=[partial(fock,4)]*n
outputarr=[partial(squeezedcubicphase,0.05, 0, 0.2)]*k #+ [partial(thermal,1)]*(n-k)

wfock = fock(4, xx, yy)
wcubic = squeezedcubicphase(0.05, 0, 0.2, xx, yy)

rl = 2*lim
from qopy.plotting import plotw
plotw([wfock, wcubic], rl)

from qopy.phase_space import measures as meas
print(meas.integrate_2d(wfock, rl))
print(meas.integrate_2d(wcubic, rl))
marg = meas.marginal(wcubic, rl)
plt.plot(marg)
plt.show()

'''
(x1,d1,i1,z1)=lorplotroutine(inputarr)
(x2,d2,i2,z2)=lorplotroutine(outputarr)

#check numerically if curves lie above and below each other
print(all(d1[0:min(np.size(d1),np.size(d2))]>=d2[0:min(np.size(d1),np.size(d2))]))
print(all(i1[0:min(np.size(i1),np.size(i2))]<=i2[0:min(np.size(i1),np.size(i2))]))


#plot curves
plt.plot(x1,d1)
plt.plot(x2,d2)

plt.figure()

plt.plot(x1,i1)
plt.plot(x2,i2)


#plot zoomed curves
plt.figure()

plt.plot(x1[0:10000],d1[0:10000])
plt.plot(x2[0:10000],d2[0:10000])

plt.figure()

plt.plot(x1[0:10000],i1[0:10000])
plt.plot(x2[0:10000],i2[0:10000])


plt.show()
'''