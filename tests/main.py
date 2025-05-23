from qopy.shortcuts import *

from qopy.phase_space.multimode import *

rl = 8
nr = 100

w1 = wig_fock(1, rl, nr)
w2 = wig_fock(2, rl, nr)

w1 = np.reshape(w1, nr ** 2)
w2 = np.reshape(w2, nr ** 2)


thresh = 1e-5
w1 = w1[np.abs(w1) < thresh]
w2 = w2[np.abs(w2) < thresh]


W = np.tensordot(w1, w2, axes=0)
W = np.reshape(W, len(w1)*len(w2))

W = W[np.abs(W) < thresh]
print(len(W))