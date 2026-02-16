import numpy as np
from scipy.integrate import simpson


def tensor(w1, w2):
    """
    Construct two-mode separable Wigner function:
    W(x1,p1,x2,p2) = W1(x1,p1) * W2(x2,p2)
    """
    return np.multiply.outer(w1, w2)


def integrate_2mode(W, rl):
    """
    Integrate a 4D two-mode Wigner function over all variables.
    """
    nr = W.shape[0]
    dx = rl / (nr - 1)

    # Successively integrate over each axis
    result = W
    for axis in reversed(range(W.ndim)):
        result = simpson(result, dx=dx, axis=axis)

    return result


def partial_trace(W, rl, mode=2):
    """
    Partial trace over one mode of a two-mode Wigner function.
    
    mode=2 → trace out (x2,p2)
    mode=1 → trace out (x1,p1)
    """
    nr = W.shape[0]
    dx = rl / (nr - 1)

    if mode == 2:
        # integrate over last two axes
        result = simpson(W, dx=dx, axis=3)
        result = simpson(result, dx=dx, axis=2)
    elif mode == 1:
        # integrate over first two axes
        result = simpson(W, dx=dx, axis=1)
        result = simpson(result, dx=dx, axis=0)
    else:
        raise ValueError("mode must be 1 or 2")

    return result