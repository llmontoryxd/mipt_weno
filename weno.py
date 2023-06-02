import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def weno_pre(xs, xp, fp, assume_sorted=False, order=4, uniform=False):
    xs = np.asarray(xs)
    xp = np.asarray(xp)
    fp = np.asarray(fp)

    # check errors
    ...

    if not assume_sorted:
        ord = np.argsort(xp)
        xp = np.ascontiguousarray(xp[ord])
        fp = np.ascontiguousarray(fp[ord])

    return weno(xs, xp, fp, order=order, uniform=uniform)


def weno(xs, xp, fp, order=4, uniform=False):
    def get_syh(xp, fp, i, p, q):
        S = np.zeros(p)
        y = np.zeros(p)
        h = np.zeros(len(S)-1)
        for s_idx in range(len(S)):
            S[s_idx] = xp[i-p+q+s_idx]
            y[s_idx] = fp[i-p+q+s_idx]
        for h_idx in range(len(h)):
            h[h_idx] = S[h_idx+1]-S[h_idx]

        return S, y, h

    def basis(x, S, k):
        l_k = [(x-S[j])/(S[k]-S[j]) for j in range(len(S)) if j != k]
        l_k = np.asarray(l_k)

        return l_k.prod()

    def pql_pattern(order):
        p = order
        q = p - 1
        l = q

        return p, q, l

    def get_beta_nonuniform(y, h, order):
        betas = np.zeros(2)
        if order == 3:
            d = np.zeros(order - 1)
            for d_idx in range(len(d)):
                d[d_idx] = (y[d_idx+1] - y[d_idx])/h[d_idx]

            dy_im1 = (2*h[0] + h[1])/(h[0]+h[1])*d[0] - h[0]/(h[0]+h[1])*d[1]
            dy_i = h[1]/(h[0]+h[1])*d[0] + h[0]/(h[0]+h[1])*d[1]
            dy_ip1 = -h[1]/(h[0]+h[1])*d[0] + (h[0] + 2*h[1])/(h[0]+h[1])*d[1]

            betas[0] = h[1]**2*(np.abs(dy_i) - np.abs(dy_im1))**2
            betas[1] = h[0]**2*(np.abs(dy_ip1) - np.abs(dy_i))**2

        if order == 4:
            H = np.sum(h)
            him = h[0]
            hi = h[1]
            hip = h[2]
            yim = y[0]
            yi = y[1]
            yip = y[2]
            yipp = y[3]

            yyim = - ((2 * him + hi) * H + him * (him + hi)) / (him * (him + hi) * H) * yim
            yyim += ((him + hi) * H) / (him * hi * (hi + hip)) * yi
            yyim -= (him * H) / ((him + hi) * hi * hip) * yip
            yyim += (him * (him + hi)) / ((hi + hip) * hip * H) * yipp

            yyi = - (hi * (hi + hip)) / (him * (him + hi) * H) * yim
            yyi += (hi * (hi + hip) - him * (2 * hi + hip)) / (him * hi * (hi + hip)) * yi
            yyi += (him * (hi + hip)) / ((him + hi) * hi * hip) * yip
            yyi -= (him * hi) / ((hi + hip) * hip * H) * yipp

            yyip = (hi * hip) / (him * (him + hi) * H) * yim
            yyip -= (hip * (him + hi)) / (him * hi * (hi + hip)) * yi
            yyip += ((him + 2 * hi) * hip - (him + hi) * hi) / ((him + hi) * hi * hip) * yip
            yyip += ((him + hi) * hi) / ((hi + hip) * hip * H) * yipp

            yyipp = - ((hi + hip) * hip) / (him * (him + hi) * H) * yim
            yyipp += (hip * H) / (him * hi * (hi + hip)) * yi
            yyipp -= ((hi + hip) * H) / ((him + hi) * hi * hip) * yip
            yyipp += ((2 * hip + hi) * H + hip * (hi + hip)) / ((hi + hip) * hip * H) * yipp

            betas[0] = (hi + hip) ** 2 * (abs(yyip - yyi) / hi - abs(yyi - yyim) / him) ** 2
            betas[1] = (him + hi) ** 2 * (abs(yyipp - yyip) / hip - abs(yyip - yyi) / hi) ** 2

        return betas

    fs = np.zeros_like(xs)
    xs_flat = xs.reshape(-1)
    fs_flat = fs.reshape(-1)
    p, q, l = pql_pattern(order)
    EPS = 10**(-6)
    alpha = 1
    pattern_changed = False

    for idx, x in enumerate(xs_flat):
        i = np.searchsorted(xp, x, side='right') - 1
        if i+q-1 >= len(xp):
            p, q, l = pql_pattern(order-1)
            i -= order-3
            pattern_changed = True

        if i-p+q < 0:
            p, q, l = pql_pattern(order-1)
            pattern_changed = True
        S, y, h = get_syh(xp, fp, i, p, q)


        m = -p+q+l
        q_ms = []
        gammas = []
        while m <= q:
            S_sub, y_sub, h_sub = get_syh(xp, fp, i, l, m)
            q_m = 0
            for k in range(len(y_sub)):
                q_m += y_sub[k]*basis(x, S_sub, k)
            q_ms.append(q_m)
            if l == q:
                if m < l:
                    gamma_m = - (x - S[-1])/(S[-1]-S[0])
                else:
                    gamma_m = (x - S[0])/(S[-1]-S[0])
            gammas.append(gamma_m)
            m += 1

        q_ms = np.asarray(q_ms)
        gammas = np.asarray(gammas)
        if order in [3, 4] and uniform is False and pattern_changed is False:
            betas = get_beta_nonuniform(y, h, order)
            alphas = gammas/(EPS+betas)**alpha
            w = alphas/(np.sum(alphas))
            fs_flat[idx] = np.sum(q_ms*w)
        else:
            fs_flat[idx] = np.sum(q_ms*gammas)

        if pattern_changed:
            p, q, l = pql_pattern(order)
            pattern_changed = False

    return fs_flat


def weno5(w, N):
    # R = 3, i = 2:(N-3)
    vmm = np.asarray(w[:, 0:N-5])
    vm = np.asarray(w[:, 1:N-4])
    vo = np.asarray(w[:, 2:N-3])
    vp = np.asarray(w[:, 3:N-2])
    vpp = np.asarray(w[:, 4:N-1])

    B0n = 13/12*(vmm-2*vm+vo)**2 + 1/4*(vmm-4*vm+3*vo)**2
    B1n = 13/12*(vm-2*vo+vp)**2 + 1/4*(vm-vp)**2
    B2n = 13/12*(vo-2*vp+vpp)**2 + 1/4*(3*vo-4*vp+vpp)**2

    d0n = 1/10
    d1n = 6/10
    d2n = 3/10
    EPS = 10**(-6)
    alpha = 2

    alpha0n = d0n/(EPS+B0n)**alpha
    alpha1n = d1n/(EPS+B1n)**alpha
    alpha2n = d2n/(EPS+B2n)**alpha

    w0n = alpha0n/(alpha0n+alpha1n+alpha2n)
    w1n = alpha1n/(alpha0n+alpha1n+alpha2n)
    w2n = alpha2n/(alpha0n+alpha1n+alpha2n)

    wn = w0n*(2*vmm-7*vm+11*vo)/6 + w1n*(-vm+5*vo+2*vp)/6+w2n*(2*vo+5*vp-vpp)/6

    umm = np.asarray(w[:, 1:N-4])
    um = np.asarray(w[:, 2:N-3])
    uo = np.asarray(w[:, 3:N-2])
    up = np.asarray(w[:, 4:N-1])
    upp = np.asarray(w[:, 5:N])

    B0p = 13/12*(umm-2*um+uo)**2 + 1/4*(umm-4*um+3*uo)**2
    B1p = 13/12*(um-2*uo+up)**2 + 1/4*(um-up)**2
    B2p = 13/12*(uo-2*up+upp)**2 + 1/4*(3*uo-4*up+upp)**2

    d0p = 3/10
    d1p = 6/10
    d2p = 1/10

    alpha0p = d0p/(EPS+B0p)**alpha
    alpha1p = d1p/(EPS+B1p)**alpha
    alpha2p = d2p/(EPS+B2p)**alpha

    w0p = alpha0p/(alpha0p+alpha1p+alpha2p)
    w1p = alpha1p/(alpha0p+alpha1p+alpha2p)
    w2p = alpha2p/(alpha0p+alpha1p+alpha2p)

    wp = w0p*(-umm+5*um+2*uo)/6 + w1p*(2*um+5*uo-up)/6 + w2p*(11*uo-7*up+2*upp)/6

    return wn, wp


def test_weno():
    N = 36
    N_int = 1000
    #seed = 14
    seed = 20
    xp = np.sort(np.random.RandomState(seed=seed).uniform(low=-1, high=1, size=N))

    def plot_test(ax, xp, fn, N_int):
        fp = fn(xp)
        xs = np.linspace(xp.min(), xp.max(), N_int)
        fs = fn(xs)

        ax.plot(xp, fp, color='black', marker='o', linestyle='None', label='Data')
        ax.plot(xs, fs, color='black', marker='None', linestyle='--', linewidth=2, label='True')

        f_weno3 = weno_pre(xs, xp, fp, order=3)
        f_weno4 = weno_pre(xs, xp, fp, order=4)
        ax.plot(xs, f_weno3, color='red', marker='None', linestyle='--', label='WENO-3 non-uniform')
        ax.plot(xs, f_weno4, color='green', marker='None', linestyle='--', label='WENO-4 non-uniform')
        ax.plot(xs, interp1d(xp, fp, kind=3)(xs), color='blue', marker='None', linestyle='--', label='Cubic spline')


    def exp(x):
        return np.exp(1.5 * x)

    def gaussian(x):
        return 5 * (1-np.exp(-4*x**2))

    def heaviside(x):
        return np.where(x < 0, 0, 4.0)

    def disc_sine(x):
        return np.where(x < 0, 2*np.sin(3*x)+4, 2*np.sin(3*x))

    fig, ax = plt.subplots(2, 2, figsize=(16, 16), constrained_layout=True)
    plot_test(ax[0, 0], xp, exp, N_int)
    plot_test(ax[0, 1], xp, gaussian, N_int)
    plot_test(ax[1, 0], xp, heaviside, N_int)
    plot_test(ax[1, 1], xp, disc_sine, N_int)
    ax[0, 0].legend()
    plt.show()

#test_weno()