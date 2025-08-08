import numpy as np
def g(n):
    if n == 0:
        return 0
    return np.log(n+1)*(n+1)-np.log(n)*n
