import numpy as np


def IC(x, IC_case):
    if IC_case == 'Sod':
        p = np.array([1, 0.1])
        u = np.array([0, 0])
        rho = np.array([1, 0.125])
    else:
        raise ValueError('Your IC is not exist')

    r0 = np.zeros(len(x))
    u0 = np.zeros(len(x))
    p0 = np.zeros(len(x))

    x_middle = (x[-1]-x[0])/2
    L = np.where(x < x_middle)
    R = np.where(x >= x_middle)

    r0[L] = rho[0]
    r0[R] = rho[1]
    u0[L] = u[0]
    u0[R] = u[1]
    p0[L] = p[0]
    p0[R] = p[1]

    return r0, u0, p0