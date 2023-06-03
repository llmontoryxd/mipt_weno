import numpy as np
import BC
import weno


def fv_ee1d(a, gamma, q, nx, dx, R, f_split, recon, x, BC_case):
    q = BC.BC(q, R, BC_case)
    #print(q)
    w = np.zeros((3, np.shape(q)[1]))
    w[0, :] = q[0, :]
    w[1, :] = q[1, :]/q[0, :]
    w[2, :] = (gamma-1)*(q[2, :] - 0.5*(q[1, :]**2)/q[0, :])

    wL, wR = recon(w, nx)

    qL = np.zeros((3, np.shape(wL)[1]))
    qR = np.zeros((3, np.shape(wR)[1]))
    qL[0, :] = wL[0, :]
    qL[1, :] = wL[1, :]*wL[0, :]
    qL[2, :] = wL[2, :]/(gamma-1)+0.5*wL[0, :]*wL[1, :]**2
    qR[0, :] = wR[0, :]
    qR[1, :] = wR[1, :] * wR[0, :]
    qR[2, :] = wR[2, :] / (gamma - 1) + 0.5 * wR[0, :] * wR[1, :] ** 2

    #

    res = np.zeros(np.shape(w))
    flux = np.zeros(np.shape(qR))
    nf = nx + 1 - 2*R

    for j in range(nf):
        flux[:, j] = f_split(qL[:, j], qR[:, j], a, gamma)

    res[:, R] = res[:, R] - flux[:, 0]/dx
    for j in range(1, nf-1):
        res[:, j+R-1] = res[:, j+R-1] + flux[:, j]/dx
        res[:, j+R] = res[:, j+R] - flux[:, j]/dx
    res[:, nx-R-1] = res[:, nx-R-1] + flux[:, nf-1]/dx

    return res