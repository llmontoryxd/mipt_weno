import numpy as np


def BC(q, R, BC_case):
    if BC_case == 'Riemann':
        nx = q.shape[1]
        #print(nx)
        for i in range(R):
            #print(i)
            q[:, i] = q[:, R]
            #print(q[:, R])
            q[:, nx-i-1] = q[:, nx-R-1]
    else:
        raise ValueError('BC has not set!')

    return q