import numpy as np
import matplotlib.pyplot as plt


def plot_IC(ax, x, res, labels):
    ax.plot(x, res, color='black')
    ax.set_xlabel('x')
    ax.set_ylabel(labels)
    ax.set_title(labels)


def IC(x, IC_case):
    if IC_case == "Sod":
        p = np.array([1, 0.1])
        u = np.array([0, 0])
        rho = np.array([1, 0.125])
    elif IC_case == "Lax":
        p = np.array([3.528, 0.571])
        u = np.array([0.698, 0])
        rho = np.array([0.445, 0.5])
    elif IC_case == "Shock_Sod":
        p = np.array([1.0, 0.1])
        u = np.array([0.75, 0])
        rho = np.array([1, 0.125])
    elif IC_case == "Supersonic":
        p = np.array([1.0, 0.02])
        u = np.array([0, 0])
        rho = np.array([1, 0.02])
    elif IC_case == "Shu_Osher":
        r0 = np.zeros(len(x))
        u0 = np.zeros(len(x))
        p0 = np.zeros(len(x))
        EPS = 10**(-6)

        x_middle = (x[-1] - x[0]) / 2
        L = np.where(x < x_middle)
        R = np.where(x >= x_middle)

        r0[L] = 3.857143
        r0[R] = 1 + EPS*np.sin(5*x[R])

        u0[L] = 2.629369
        u0[R] = 0

        p0[L] = 10.3333
        p0[R] = 1

        fig, ax = plt.subplots(1, 3, figsize=(32, 8), constrained_layout=True)
        labels = np.array(['Density', 'Velocity', 'Pressure'])

        plot_IC(ax[0], x, r0, labels[0])
        plot_IC(ax[1], x, u0, labels[1])
        plot_IC(ax[2], x, p0, labels[2])

        filename_str = IC_case
        plt.savefig('outData/' + filename_str + '.png')

        return r0, u0, p0

    else:
        raise ValueError('Your IC is not exist')

    fig, ax = plt.subplots(1, 3, figsize=(32, 8), constrained_layout=True)
    labels = np.array(['Density', 'Velocity', 'Pressure'])

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

    plot_IC(ax[0], x, r0, labels[0])
    plot_IC(ax[1], x, u0, labels[1])
    plot_IC(ax[2], x, p0, labels[2])

    filename_str = IC_case
    plt.savefig('outData/' + filename_str + '.png')

    return r0, u0, p0