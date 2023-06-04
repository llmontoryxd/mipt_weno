import numpy as np
import matplotlib.pyplot as plt
import weno
import flux_splitting
import IC
import exact_solution
import solver_r


def plot_res(ax, xe, rese, x, res, labels):
    ax.plot(xe, rese, linestyle='--', marker='None', color='black', label='exact')
    ax.plot(x, res, linestyle='--', marker='None', color='red', label='numerical')
    ax.set_ylabel(labels)
    ax.set_xlabel('x')
    ax.set_title(labels)
    ax.legend(loc='best')


def read_config(filename):
    f_split_dict = {
        'HLLE': flux_splitting.HLLE,
        'HLLC': flux_splitting.HLLC,
        'LF': flux_splitting.LF,
        'Rusanov': flux_splitting.Rusanov
    }

    recon_dict = {
        'weno3': weno.weno3,
        'weno5': weno.weno5,
        'weno7': weno.weno7
    }

    solv_dict = {
        'fv1d': solver_r.fv_ee1d
    }

    with open(filename, 'r', encoding='utf-8') as file:
        for n_line, line in enumerate(file):
            if n_line == 0:
                L0 = float(line.split(' ')[1])
            elif n_line == 1:
                Lx = float(line.split(' ')[1])
            elif n_line == 2:
                CFL = float(line.split(' ')[1])
            elif n_line == 3:
                t_f = float(line.split(' ')[1])
            elif n_line == 4:
                nx = int(line.split(' ')[1])
            elif n_line == 5:
                gamma = float(line.split(' ')[1])
            elif n_line == 6:
                IC_case = str(line.split(' ')[1]).replace('\n', '')
            elif n_line == 7:
                BC_case = str(line.split(' ')[1]).replace('\n', '')
            elif n_line == 8:
                f_split_str = str(line.split(' ')[1]).replace('\n', '')
                f_split = f_split_dict[f_split_str]
            elif n_line == 9:
                recon_str = str(line.split(' ')[1]).replace('\n', '')
                recon = recon_dict[recon_str]
            elif n_line == 10:
                solv_str = str(line.split(' ')[1]).replace('\n', '')
                solv = solv_dict[solv_str]

    return L0, Lx, CFL, t_f, nx, gamma, IC_case, BC_case, f_split, recon, solv


def solve(filename):
    L0, Lx, CFL, t_f, nx, gamma, IC_case, BC_case, f_split, recon, solv = read_config(filename)
    plot_every_step = False


    dx = (Lx - L0) / nx
    xc = np.linspace(L0, Lx, nx + 1)
    xc += dx / 2
    xc = np.delete(xc, np.where(xc > Lx))

    r0, u0, p0 = IC.IC(xc, IC_case)
    E0 = p0 / (gamma - 1) + 0.5 * r0 * u0 ** 2
    a0 = np.sqrt(gamma * p0 / r0)
    Q0 = np.array([r0, r0 * u0, E0])

    if recon == weno.weno3:
        R = 2
    elif recon == weno.weno5:
        R = 3
    elif recon == weno.weno7:
        R = 4
    else:
        raise ValueError('Please, choose right recon')

    nx += 2 * R
    q0 = np.zeros([3, nx])
    q0[:, R:(nx - R)] = Q0

    lambda0 = np.max(np.abs(u0) + a0)
    dt0 = CFL * dx / lambda0

    q = q0
    it = 0
    dt = dt0
    t = 0
    lamda = lambda0

    while t < t_f:
        if t + dt > t_f:
            dt = t_f - t
        t = t + dt

        q0 = q

        L = solv(lamda, gamma, q, nx, dx, R, f_split, recon, xc, BC_case)
        q = q0 - dt * L

        L = solv(lamda, gamma, q, nx, dx, R, f_split, recon, xc, BC_case)
        q = 0.75 * q0 + 0.25 * (q - dt * L)

        L = solv(lamda, gamma, q, nx, dx, R, f_split, recon, xc, BC_case)
        q = (q0 + 2 * (q - dt * L)) / 3



        r = q[0, :]
        u = q[1, :] / r
        E = q[2, :]
        p = (gamma - 1) * (E - 0.5 * r * u ** 2)
        a = np.sqrt(gamma * p / r)

        lamda = np.max(np.abs(u) + a)
        dt = CFL * dx / lamda
        it += 1

        xe, rhoe, uxe, pe, e, temp, Mache, entro = exact_solution.EulerExact(gamma, np.array([r0[0], r0[-1]]),
                                                                             np.array([u0[0], u0[-1]]),
                                                                             np.array([p0[0], p0[-1]]), t, xc)

        if plot_every_step or (t == t_f and plot_every_step is False):
            title_str = f'L0 = {L0}, Lx = {Lx}, CFL = {CFL}, Final time = {t_f},'+\
                        f'nx = {nx}, gamma = {gamma}\n' +\
                        f'IC = {IC_case}, BC = {BC_case}\n' +\
                        f'flux = {f_split.__name__}, recon = {recon.__name__}, solver = {solv.__name__}'
            filename_str = f'{L0} {Lx} {CFL} {t_f} {nx} {gamma} {IC_case} {BC_case} {f_split.__name__} ' +\
                           f'{recon.__name__} {solv.__name__}'
            fig, ax = plt.subplots(1, 3, figsize=(32, 8), constrained_layout=True)
            labels = ['Density', 'Velocity', 'Pressure', 'Entropy', 'Mach number', 'Internal energy']
            plot_res(ax[0], xe, rhoe, xc, r[R:nx - R], labels[0])
            plot_res(ax[1], xe, uxe, xc, u[R:nx - R], labels[1])
            plot_res(ax[2], xe, pe, xc, p[R:nx - R], labels[2])
            fig.suptitle(title_str)
            plt.savefig('outData/' + filename_str + '.png')

            return xc, nx, r[R:nx-R], u[R:nx - R], p[R:nx-R]
