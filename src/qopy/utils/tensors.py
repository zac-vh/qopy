import numpy as np
import scipy




def tensor(A, B):
    return np.tensordot(A, B, axes=0)