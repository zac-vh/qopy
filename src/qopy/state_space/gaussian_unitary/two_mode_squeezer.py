import numpy as np
from qopy.state_space.gaussian_unitary import beamsplitter



def transition_amplitude(i, k, n, g=2):
    # Amplitude of transition in a TMS
    # tms_ikn = <n,m|Utms_lba|i,k> with n-m=i-kxs
    return beamsplitter.transition_amplitude(i, n + k - i, n, 1 / g) / np.sqrt(g)
