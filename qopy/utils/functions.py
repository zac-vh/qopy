import numpy as np
from scipy.optimize import brentq

def g(n):
    n = np.asarray(n, dtype=float)

    return np.where(
        n == 0,
        0.0,
        np.log1p(n) + n * np.log1p(1 / n)
    )


def ginv(y):
    def inverse(value):
        if value == 0:
            return 0.0

        upper = 1.0
        while g(upper) < value:
            upper *= 2.0

        return brentq(
            lambda n: g(n) - value,
            0.0,
            upper,
            xtol=5e-324,
            rtol=4 * np.finfo(float).eps
        )

    y = np.asarray(y, dtype=float)
    result = np.vectorize(inverse, otypes=[float])(y)

    return result.item() if y.ndim == 0 else result