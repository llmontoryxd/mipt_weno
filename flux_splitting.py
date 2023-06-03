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


def HLLC(qL, qR, smax, gamma):
    rl = qL[0]
    ul = qL[1] / rl
    El = qL[2] / rl
    pl = (gamma - 1) * (qL[2] - rl * ul ** 2 / 2)
    al = np.sqrt(gamma * pl / rl)

    rr = qR[0]
    ur = qR[1] / rr
    Er = qR[2] / rr
    pr = (gamma - 1) * (qR[2] - rr * ur ** 2 / 2)
    ar = np.sqrt(gamma * pr / rr)

    FL = np.array([rl * ul, rl * ul ** 2 + pl, ul * (rl * El + pl)])
    FR = np.array([rr * ur, rr * ur ** 2 + pr, ur * (rr * Er + pr)])

    PPV = max(0, 0.5*(pl+pr)+0.5*(ul-ur)*(0.25*(rl+rr)*(al+ar)))
    pmin = min(pl, pr)
    pmax = max(pl, pr)
    Qmax = pmax/pmin
    Quser = 2.0

    if (Qmax <= Quser) and (pmin <= PPV) and (PPV <= pmax):
        pM = PPV
    else:
        if PPV < pmin:
            PQ = (pl/pr)**(gamma-1)/(2*gamma)
            uM = (PQ*ul/al + ur/ar + 2/(gamma-1)*(PQ-1))/(PQ/al+1/ar)
            PTL = 1 + (gamma-1)/2*(ul-uM)/al
            PTR = 1 + (gamma-1)/2*(uM-ur)/ar
            pM = 0.5*(pl*PTL**(2*gamma/(gamma-1)) + pr*PTR**(2*gamma/(gamma-1)))
        else:
            GEL = np.sqrt((2/(gamma+1)/rl)/((gamma-1)/(gamma+1)*pl + PPV))
            GER = np.sqrt((2/(gamma+1)/rr)/((gamma-1)/(gamma+1)*pr + PPV))
            pM = (GEL*pl + GER*pr - (ur-ul))/(GER+GEL)

    if pM > pl:
        zL = np.sqrt(1+(gamma+1)/(2*gamma)*(pM/pl - 1))
    else:
        zL = 1

    if pM > pr:
        zR = np.sqrt(1+(gamma+1)/(2*gamma)*(pM/pr - 1))
    else:
        zR = 1

    SL = ul - al*zL
    SR = ur + ar*zR
    SM = (pl-pr + rr*ur*(SR-ur) - rl*ul*(SL-ul))/(rr*(SR-ur) - rl*(SL-ul))

    if 0 <= SL:
        flux = FL
    elif (SL <= 0) and (0 <= SM):
        qsl = rl*(SL-ul)/(SL-SM)*np.array([1, SM, qL[2]/rl + (SM-ul)*(SM + pl/(rl*(SL-ul)))])
        flux = FL + SL*(qsl - qL)
    elif (SM <= 0) and (0 <= SR):
        qsr = rr*(SR-ur)/(SR-SM)*np.array([1, SM, qR[2]/rr + (SM-ur)*(SM + pr/(rr*(SR-ur)))])
        flux = FR + SR * (qsr - qR)
    elif 0 >= SR:
        flux = FR
    else:
        raise ValueError('Cannot compute flux')

    return flux
