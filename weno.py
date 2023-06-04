import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def weno_pre(xs, xp, fp, assume_sorted=False, order=4, uniform=False):
    xs = np.asarray(xs)
    xp = np.asarray(xp)
    fp = np.asarray(fp)

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


def weno3(w, N):
    # R = 2, i = 1:(N-2)
    vm = np.asarray(w[:, 0:N-3])
    vo = np.asarray(w[:, 1:N-2])
    vp = np.asarray(w[:, 2:N-1])

    #B0n = (vo - vm)**2
    #B1n = (vp - vo)**2
    B0n = 1/4*(np.abs(vp - vm) - np.abs(4*vo - 3*vm - vp))**2
    B1n = 1/4*(np.abs(vp - vm) - np.abs(4*vo - vm - 3*vp))**2


    d0n = 1/3
    d1n = 2/3

    EPS = 10**(-6)
    alpha = 2

    alpha0n = d0n/(EPS+B0n)**alpha
    alpha1n = d1n/(EPS+B1n)**alpha
    alphasumn = alpha0n+alpha1n

    w0n = alpha0n/alphasumn
    w1n = alpha1n/alphasumn

    wn = w0n*(-vm + 3*vo)/2 + w1n*(vo + vp)/2

    um = np.asarray(w[:, 1:N-2])
    uo = np.asarray(w[:, 2:N-1])
    up = np.asarray(w[:, 3:N])

    #B0p = (uo - um)**2
    #B1p = (up - uo)**2
    B0p = 1 / 4 * (np.abs(up - um) - np.abs(4 * uo - 3 * um - up)) ** 2
    B1p = 1 / 4 * (np.abs(up - um) - np.abs(4 * uo - um - 3 * up)) ** 2

    d0p = 2/3
    d1p = 1/3

    alpha0p = d0p / (EPS + B0p) ** alpha
    alpha1p = d1p / (EPS + B1p) ** alpha
    alphasump = alpha0p + alpha1p

    w0p = alpha0p / alphasump
    w1p = alpha1p / alphasump

    wp = w0p*(uo+um)/2 + w1p*(-up+3*uo)/2

    return wn, wp


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


def weno7(w, N):
    def get_smooth_indicators(fmmm, fmm, fm, fo, fp, fpp, fppp):
        B0 = fmmm*(547*fmmm - 3882*fmm + 4642*fm - 1854*fo) +\
             fmm*(7043*fmm - 17246*fm + 7042*fo) +\
             fm*(11003*fm - 9402*fo) + 2107*fo**2

        B1 = fmm*(267*fmm - 1642*fm + 1602*fo - 494*fp) +\
             fm*(2843*fm - 5966*fo + 1922*fp) +\
             fo*(3443*fo - 2522*fp) + 547*fp**2

        B2 = fm*(547*fm - 2522*fo + 1922*fp - 494*fpp) +\
             fo*(3443*fo - 5966*fp + 1602*fpp) +\
             fp*(2843*fp - 1642*fpp) + 267*fpp**2

        B3 = fo*(2107*fo - 9402*fp + 7042*fpp - 1854*fppp) +\
             fp*(11003*fp - 17246*fpp + 4642*fppp) +\
             fpp*(7043*fpp - 3882*fppp) + 547*fppp**2

        return B0, B1, B2, B3

    def get_alpha(g, tau, eps, alpha, beta):
        return g*(1+(tau)**alpha/(beta+eps)**alpha)

    # R = 4, i = 3:(N-4)
    vmmm = np.asarray(w[:, 0:N-7])
    vmm = np.asarray(w[:, 1:N-6])
    vm = np.asarray(w[:, 2:N-5])
    vo = np.asarray(w[:, 3:N-4])
    vp = np.asarray(w[:, 4:N-3])
    vpp = np.asarray(w[:, 5:N-2])
    vppp = np.asarray(w[:, 6:N-1])


    #B0n = vm * (134241 * vm - 114894 * vo) + vmmm * (56694 * vm - 47214 * vmm + 6649 * vmmm - 22778 * vo) +\
    #    25729 * vo ** 2 + vmm * (-210282 * vm + 85641 * vmm + 86214 * vo)
    #B1n = vo * (41001 * vo - 30414 * vp) + vmm * (-19374 * vm + 3169 * vmm + 19014 * vo - 5978 * vp) +\
    #    6649 * vp ** 2 + vm * (33441 * vm - 70602 * vo + 23094 * vp)
    #B2n = vp * (33441 * vp - 19374 * vpp) + vm * (6649 * vm - 30414 * vo + 23094 * vp - 5978 * vpp) +\
    #    3169 * vpp ** 2 + vo * (41001 * vo - 70602 * vp + 19014 * vpp)
    #B3n = vpp * (85641 * vpp - 47214 * vppp) + vo * (25729 * vo - 114894 * vp + 86214 * vpp - 22778 * vppp) +\
    #    6649 * vppp ** 2 + vp * (134241 * vp - 210282 * vpp + 56694 * vppp)

    B0n, B1n, B2n, B3n = get_smooth_indicators(vmmm, vmm, vm, vo, vp, vpp, vppp)



    g0 = 1/35
    g1 = 12/35
    g2 = 18/35
    g3 = 4/35

    EPS = 10**(-10)
    alpha = 4
    tau = np.abs(B0n - B3n)

    alpha0n = g0 / (EPS + B0n) ** alpha
    alpha1n = g1 / (EPS + B1n) ** alpha
    alpha2n = g2 / (EPS + B2n) ** alpha
    alpha3n = g3 / (EPS + B3n) ** alpha

    #alpha0n = get_alpha(g0, tau, EPS, alpha, B0n)
    #alpha1n = get_alpha(g1, tau, EPS, alpha, B1n)
    #alpha2n = get_alpha(g2, tau, EPS, alpha, B2n)
    #alpha3n = get_alpha(g3, tau, EPS, alpha, B3n)
    alphasummn = alpha0n+alpha1n+alpha2n+alpha3n

    w0n = alpha0n / alphasummn
    w1n = alpha1n / alphasummn
    w2n = alpha2n / alphasummn
    w3n = alpha3n / alphasummn

    wn = w0n*(-3*vmmm + 13*vmm - 23*vm + 25*vo)/12 + w1n*(1*vmm - 5*vm + 13*vo + 3*vp)/12 +\
        w2n*(-1*vm + 7*vo + 7*vp - 1*vpp)/12 + w3n*(3*vo + 13*vp - 5*vpp + 1*vppp)/12



    ummm = np.asarray(w[:, 1:N - 6])
    umm = np.asarray(w[:, 2:N - 5])
    um = np.asarray(w[:, 3:N - 4])
    uo = np.asarray(w[:, 4:N - 3])
    up = np.asarray(w[:, 5:N - 2])
    upp = np.asarray(w[:, 6:N - 1])
    uppp = np.asarray(w[:, 7:N])


    #B0p = um * (134241 * um - 114894 * uo) + ummm * (56694 * um - 47214 * umm + 6649 * ummm - 22778 * uo) +\
    #    25729 * uo ** 2 + umm * (-210282 * um + 85641 * umm + 86214 * uo)
    #B1p = uo * (41001 * uo - 30414 * up) + umm * (-19374 * um + 3169 * umm + 19014 * uo - 5978 * up) +\
    #    6649 * up ** 2 + um * (33441 * um - 70602 * uo + 23094 * up)
    #B2p = up * (33441 * up - 19374 * upp) + um * (6649 * um - 30414 * uo + 23094 * up - 5978 * upp) +\
    #    3169 * upp ** 2 + uo * (41001 * uo - 70602 * up + 19014 * upp)
    #B3p = upp * (85641 * upp - 47214 * uppp) + uo * (25729 * uo - 114894 * up + 86214 * upp - 22778 * uppp) +\
    #    6649 * uppp ** 2 + up * (134241 * up - 210282 * upp + 56694 * uppp)

    B0p, B1p, B2p, B3p = get_smooth_indicators(ummm, umm, um, uo, up, upp, uppp)

    g0 = 4/35
    g1 = 18/35
    g2 = 12/35
    g3 = 1/35

    tau = np.abs(B0p - B3p)

    alpha0p = g0 / (EPS + B0p) ** alpha
    alpha1p = g1 / (EPS + B1p) ** alpha
    alpha2p = g2 / (EPS + B2p) ** alpha
    alpha3p = g3 / (EPS + B3p) ** alpha

    #alpha0p = get_alpha(g0, tau, EPS, alpha, B0p)
    #alpha1p = get_alpha(g1, tau, EPS, alpha, B1p)
    #alpha2p = get_alpha(g2, tau, EPS, alpha, B2p)
    #alpha3p = get_alpha(g3, tau, EPS, alpha, B3p)
    alphasummp = alpha0p + alpha1p + alpha2p + alpha3p

    w0p = alpha0p / alphasummp
    w1p = alpha1p / alphasummp
    w2p = alpha2p / alphasummp
    w3p = alpha3p / alphasummp

    wp = w0p*(1*ummm - 5*umm + 13*um + 3*uo)/12 + w1p*(-1*umm + 7*um + 7*uo - 1*up)/12 +\
        w2p*(3*um + 13*uo - 5*up + 1*upp)/12 + w3p*(25*uo - 23*up + 13*upp - 3*uppp)/12

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
