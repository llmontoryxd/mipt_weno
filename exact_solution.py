import numpy as np
from scipy.optimize import fsolve
from math import sqrt


def EulerExact(gamma, rho, u, p, t_f, x):
    alpha = (gamma+1)/(gamma-1)

    PRL = p[1]/p[0]
    cright = sqrt(gamma*p[1]/rho[1])
    cleft = sqrt(gamma*p[0]/rho[0])
    CRL = cright/cleft
    MACHLEFT = (u[0]-u[1])/cleft

    f = lambda P: (1+MACHLEFT*(gamma-1)/2-(gamma-1)*CRL*(P-1)/sqrt(2*gamma*(gamma-1+(gamma+1)*P)))**(2*gamma/(gamma-1))/P-PRL
    p34 = fsolve(f, 3)

    p3 = p34*p[1]
    rho3 = rho[1]*(1+alpha*p34)/(alpha+p34)
    rho2 = rho[0]*(p34*PRL)**(1/gamma)
    u2 = u[0] - u[1] + (2/(gamma-1))*cleft*(1-(p34*PRL)**((gamma-1)/(2*gamma)))
    c2 = sqrt(gamma*p3/rho2)
    spos = 0.5 + t_f*cright*sqrt((gamma-1)/(2*gamma) + (gamma+1)/(2*gamma)*p34)+t_f*u[1]

    x0 = (x[-1]-x[0])/2
    conpos = x0 + u2*t_f + t_f*u[1]
    pos1 = x0 + (u[0] - cleft)*t_f
    pos2 = x0 + (u2 + u[1]-c2)*t_f

    pe = np.zeros(len(x))
    uxe = np.zeros(len(x))
    rhoe = np.zeros(len(x))
    Mache = np.zeros(len(x))
    ce = np.zeros(len(x))

    for i in range(len(x)):
        if x[i] <= pos1:
            pe[i] = p[0]
            rhoe[i] = rho[0]
            uxe[i] = u[0]
        elif x[i] <= pos2:
            pe[i] = p[0]*(1+(pos1-x[i])/(cleft*alpha*t_f))**(2*gamma/(gamma-1))
            rhoe[i] = rho[0]*(1+(pos1-x[i])/(cleft*alpha*t_f))**(2/(gamma-1))
            uxe[i] = u[0] + (2/(gamma+1))*(x[i]-pos1)/t_f
        elif x[i] <= conpos:
            pe[i] = p3
            rhoe[i] = rho2
            uxe[i] = u2+u[1]
        elif x[i] <= spos:
            pe[i] = p3
            rhoe[i] = rho3
            uxe[i] = u2+u[1]
        else:
            pe[i] = p[1]
            rhoe[i] = rho[1]
            uxe[i] = u[1]

        ce[i] = sqrt(gamma * pe[i] / rhoe[i])
        Mache[i] = uxe[i] / ce[i]

    entro = np.log(pe/rhoe**gamma)
    e = pe/((gamma-1)*rhoe)
    temp = pe/rhoe

    return x, rhoe, uxe, pe, e, temp, Mache, entro



