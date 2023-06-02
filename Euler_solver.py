import numpy as np
import matplotlib.pyplot as plt
import weno
import flux_splitting
import IC
import exact_solution
import solver_r
import BC


def plot_res(ax, xe, rese, x, res, labels):
    ax.plot(xe, rese, linestyle='--', marker='None', color='black', label='exact')
    ax.plot(x, res, linestyle='--', marker='None', color='red', label='numerical')
    ax.set_ylabel(labels)
    ax.set_xlabel('x')
    ax.set_title(labels)
    ax.legend(loc='best')


L0 = 0.0
Lx = 1.0
CFL = 0.55
t_f = 0.12
nx = 200
gamma = 1.4
IC_case = 'Sod'
BC_case = 'Riemann'
f_split = flux_splitting.HLLE
recon = weno.weno5
solv = solver_r.fd_ee1d
plot_every_step = False

#
dx = (Lx-L0)/nx
xc = np.linspace(L0, Lx, nx+1)
xc += dx/2
xc = np.delete(xc, np.where(xc > Lx))


#
r0, u0, p0 = IC.IC(xc, IC_case)
E0 = p0/(gamma-1) + 0.5*r0*u0**2
a0 = np.sqrt(gamma*p0/r0)
Q0 = np.array([r0, r0*u0, E0])

xe, rhoe, uxe, pe, e, temp, Mache, entro = exact_solution.EulerExact(gamma, np.array([r0[0], r0[-1]]),
                                                                     np.array([u0[0], u0[-1]]),
                                                                     np.array([p0[0], p0[-1]]), t_f, xc)

if recon == weno.weno5:
    R = 3
elif recon == weno.weno_pre:
    order = 4
    R = 2
else:
    raise ValueError('Please, choose right recon')
nx += 2*R
q0 = np.zeros([3, nx])
q0[:, R:(nx-R)] = Q0
#print(q0)

lambda0 = np.max(np.abs(u0)+a0)
dt0 = CFL*dx/lambda0

q = q0
it = 0
dt = dt0
t = 0
lamda = lambda0

while t < t_f:
    if t+dt > t_f:
        dt = t_f-t
    t = t + dt

    q0 = q

    L = solv(lamda, gamma, q, nx, dx, R, f_split, recon, xc, BC_case)
    q = q0 - dt*L

    L = solv(lamda, gamma, q, nx, dx, R, f_split, recon, xc, BC_case)
    q = 0.75*q0 + 0.25*(q-dt*L)

    L = solv(lamda, gamma, q, nx, dx, R, f_split, recon, xc, BC_case)
    q = (q0+2*(q-dt*L))/3


    r = q[0, :]
    u = q[1, :]/r
    E = q[2, :]
    p = (gamma-1)*(E-0.5*r*u**2)
    a = np.sqrt(gamma*p/r)

    lamda = np.max(np.abs(u) + a)
    dt = CFL*dx/lamda
    it += 1

    xe, rhoe, uxe, pe, e, temp, Mache, entro = exact_solution.EulerExact(gamma, np.array([r0[0], r0[-1]]),
                                                                         np.array([u0[0], u0[-1]]),
                                                                         np.array([p0[0], p0[-1]]), t, xc)

    if plot_every_step or (t == t_f and plot_every_step is False):
        fig, ax = plt.subplots(1, 3, figsize=(32, 16), constrained_layout=True)
        labels = ['Density', 'Velocity', 'Pressure', 'Entropy', 'Mach number', 'Internal energy']
        plot_res(ax[0], xe, rhoe, xc, r[R:nx-R], labels[0])
        plot_res(ax[1], xe, uxe, xc, u[R:nx-R], labels[1])
        plot_res(ax[2], xe, pe, xc, p[R:nx-R], labels[2])
        plt.show()


xe, rhoe, uxe, pe, e, temp, Mache, entro = exact_solution.EulerExact(gamma, np.array([r0[0], r0[-1]]),
                                                                     np.array([u0[0], u0[-1]]),
                                                                     np.array([p0[0], p0[-1]]), t_f, xc)

#fig, ax = plt.subplots(2, 3, figsize=(32, 16), constrained_layout=True)
#labels = ['Density', 'Velocity', 'Pressure', 'Entropy', 'Mach number', 'Internal energy']
#plot_res(ax[0, 0], xe, rhoe, labels[0])
#plot_res(ax[0, 1], xe, uxe, labels[1])
#plot_res(ax[0, 2], xe, pe, labels[2])
#plot_res(ax[1, 0], xe, entro, labels[3])
#plot_res(ax[1, 1], xe, Mache, labels[4])
#plot_res(ax[1, 2], xe, e, labels[5])
#plt.show()
