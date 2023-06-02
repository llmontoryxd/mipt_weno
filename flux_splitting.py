import numpy as np


def LF(qL, qR, smax, gamma):
    rl = qL[0]
    ul = qL[1]/rl
    pl = (gamma-1)*(qL[2] - rl*ul**2/2)
    hl = (qL[2] + pl)/rl

    rr = qR[0]
    ur = qR[1] / rr
    pr = (gamma - 1) * (qR[2] - rr * ur ** 2 / 2)
    hr = (qR[2] + pr) / rr

    FL = np.array([rl*ul, rl*ul**2+pl, ul*(rl*hl)])
    FR = np.array([rr*ur, rr*ur**2+pr, ur*(rr*hr)])

    flux = 0.5*(FR + FL + smax*(qL-qR))

    return flux


def Rusanov(qL, qR, smax, gamma):
    rl = qL[0]
    ul = qL[1] / rl
    El = qL[2] / rl
    pl = (gamma - 1) * (qL[2] - rl * ul ** 2 / 2)
    hl = (qL[2] + pl) / rl

    rr = qR[0]
    ur = qR[1] / rr
    Er = qR[2] / rr
    pr = (gamma - 1) * (qR[2] - rr * ur ** 2 / 2)
    hr = (qR[2] + pr) / rr

    #
    RT = np.sqrt(rr/rl)
    u = (ul + RT*ur)/(1+RT)
    H = (hl + RT*hr)/(1+RT)
    a = np.sqrt((gamma-1)*(H-u**2/2))

    FL = np.array([rl * ul, rl * ul ** 2 + pl, ul * (rl * El + pl)])
    FR = np.array([rr * ur, rr * ur ** 2 + pr, ur * (rr * Er + pr)])

    ssmax = np.abs(u) + a

    flux = 0.5*(FR + FL + ssmax*(qL-qR))

    return flux


def Roe(qL, qR, smax, gamma):
    rl = qL[0]
    ul = qL[1] / rl
    El = qL[2] / rl
    pl = (gamma - 1) * (qL[2] - rl * ul ** 2 / 2)
    al = np.sqrt(gamma*pl/rl)
    hl = (qL[2] + pl) / rl

    rr = qR[0]
    ur = qR[1] / rr
    Er = qR[2] / rr
    pr = (gamma - 1) * (qR[2] - rr * ur ** 2 / 2)
    ar = np.sqrt(gamma*pr/rr)
    hr = (qR[2] + pr) / rr

    RT = np.sqrt(rr / rl)
    r = RT*rl
    u = (ul + RT * ur) / (1 + RT)
    H = (hl + RT * hr) / (1 + RT)
    a = np.sqrt((gamma - 1) * (H - u ** 2 / 2))

    dr = rr - rl
    du = ur - ul
    dp = pr - pl

    dV = np.array([(dp-r*a*du)/(2*a**2), -(dp/(a**2)-dr), (dp+r*a*du)/(2*a**2)])
    ws = np.array([np.abs(u-a), np.abs(u), np.abs(u+a)])

    Da = np.max(0.4*((ur-ar) - (ul-al)))
    if ws[0] < Da/2:
        ws[0] = ws[0]*ws[0]/Da + Da/4

    Da = np.max(0.4 * ((ur + ar) - (ul + al)))
    if ws[2] < Da / 2:
        ws[2] = ws[2] * ws[2] / Da + Da / 4

    R = np.array([[1, 1, 1], [u-a, u, u+a], [H-u*a, u**2/2, H+u*a]])

    FL = np.array([rl * ul, rl * ul ** 2 + pl, ul * (rl * El + pl)])
    FR = np.array([rr * ur, rr * ur ** 2 + pr, ur * (rr * Er + pr)])

    flux = (FL + FR - np.prod(R, (ws*dV)))/2

    return flux



def HLLE(qL, qR, smax, gamma):
    rl = qL[0]
    ul = qL[1] / rl
    El = qL[2] / rl
    pl = (gamma - 1) * (qL[2] - rl * ul ** 2 / 2)
    al = np.sqrt(gamma * pl / rl)
    hl = (qL[2] + pl) / rl

    rr = qR[0]
    ur = qR[1] / rr
    Er = qR[2] / rr
    pr = (gamma - 1) * (qR[2] - rr * ur ** 2 / 2)
    ar = np.sqrt(gamma * pr / rr)
    hr = (qR[2] + pr) / rr

    RT = np.sqrt(rr / rl)
    u = (ul + RT * ur) / (1 + RT)
    H = (hl + RT * hr) / (1 + RT)
    a = np.sqrt((gamma - 1) * (H - u ** 2 / 2))

    SLm = min(ul-al, u-a)
    SRp = max(ur+ar, u+a)

    FL = np.array([rl * ul, rl * ul ** 2 + pl, ul * (rl * El + pl)])
    FR = np.array([rr * ur, rr * ur ** 2 + pr, ur * (rr * Er + pr)])

    if 0 <= SLm:
        flux = FL
    elif SLm <= 0 and 0 <= SRp:
        flux = (SRp*FL - SLm*FR + SLm*SRp*(qR-qL))/(SRp-SLm)
    else:
        flux = FR

    return flux