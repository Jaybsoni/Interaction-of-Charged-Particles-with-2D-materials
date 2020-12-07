import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.special as spec
import matplotlib.pyplot as plt

# Constants: ###########################################################################################################
n_0 = 0.428
alpha = np.pi * n_0
e_s = 3.9
z_0 = 3
Q = 1
L = (2 * np.pi * n_0) / alpha
# h = 4
# gamma = 0.1

# Integration functions for F_image and F_stopping: ####################################################################


def chi(kx, ky, vx, gamma):
    k_sqrd = kx**2 + ky**2

    numerator = k_sqrd * n_0
    denominator = alpha*k_sqrd - (kx*vx)**2 - np.complex(0, 1)*gamma*(kx*vx)

    return numerator / denominator


def e_0(kx, ky, h):
    k_sqrd = kx**2 + ky**2
    k = np.sqrt(k_sqrd)

    frac = (e_s - 1)/(e_s + 1)
    denominator = 1 - frac*np.exp(-2*k*h)

    return 1 / denominator


def e(kx, ky, vx, gamma, h):
    k_sqrd = kx**2 + ky**2
    k = np.sqrt(k_sqrd)

    return e_0(kx, ky, h) + chi(kx, ky, vx, gamma) * ((2*np.pi) / k)


def f_stp_integrand(kx, ky, vx, gamma, h):
    k_sqrd = kx**2 + ky**2
    k = np.sqrt(k_sqrd)

    pre_factor = kx * np.exp(-2 * k * z_0) / k
    complex_func = np.imag(1 / e(kx, ky, vx, gamma, h))
    if type(complex_func) not in (int, float, np.float64):
        print('Error, type real_func = {}'.format(type(complex_func)))
    integrand = pre_factor * complex_func

    return integrand


def f_stp(vx, gamma, h):
    pre_factor = 2 * (Q ** 2) / np.pi
    result = integrate.dblquad(lambda dkx, dky: f_stp_integrand(dkx, dky, vx, gamma, h),
                               0, np.inf, lambda t: 0, lambda t: np.inf, epsabs=1.49e-04, epsrel=1.49e-04)

    return pre_factor * result[0]


def f_im_integrand(kx, ky, vx, gamma, h):
    k_sqrd = kx ** 2 + ky ** 2
    k = np.sqrt(k_sqrd)

    pre_factor = np.exp(-2 * k * z_0)
    real_func = np.real(1 - (1 / e(kx, ky, vx, gamma, h)))
    if type(real_func) not in (int, float, np.float64):
        print('Error, type real_func = {}'.format(type(real_func)))
    integrand = pre_factor * real_func

    return integrand


def f_im(vx, gamma, h):
    pre_factor = -2 * (Q ** 2) / np.pi
    result = integrate.dblquad(lambda dkx, dky: f_im_integrand(dkx, dky, vx, gamma, h),
                               0, np.inf, lambda t: 0, lambda t: np.inf, epsabs=1.49e-04, epsrel=1.49e-04)

    return pre_factor * result[0]


# Integration functions for KKR based expressions of F_image and F_stopping: ###########################################


def get_f_im_0(h):
    const = -1 * (Q ** 2) / (4 * (z_0 + h)**2)
    const2 = (e_s - 1) / (e_s + 1)
    return const * const2


def chi_kkr(k, w, gamma):
    k_sqrd = k**2

    numerator = k_sqrd * n_0
    denominator = alpha*k_sqrd - w**2 - np.complex(0, 1)*gamma*w

    return numerator / denominator


def e_0_kkr(k, h):
    frac = (e_s - 1)/(e_s + 1)
    denominator = 1 - frac*np.exp(-2*k*h)

    return 1 / denominator


def e_kkr(k, w, gamma, h):
    return e_0_kkr(k, h) + chi_kkr(k, w, gamma) * ((2*np.pi) / k)


def f_im_integrand_kkr(k, w, v, gamma, h):

    pre_factor_k = k * np.exp(-2 * k * z_0)
    pre_factor_w = 1 / np.sqrt(w**2 - (k*v)**2)
    im_func = np.imag(1 / e_kkr(k, w, gamma, h))
    if type(im_func) not in (int, float, np.float64):
        print('Error, type real_func = {}'.format(type(im_func)))
    integrand = pre_factor_k * pre_factor_w * im_func

    return integrand


def f_im_kkr(v, gamma, h):
    pre_factor = 2 * (Q ** 2) / np.pi
    result = integrate.dblquad(lambda dw, dk: f_im_integrand_kkr(dk, dw, v, gamma, h),
                               0, np.inf, lambda k: k*v, lambda k: np.inf, epsabs=1.49e-04, epsrel=1.49e-04)
    f_im_0 = get_f_im_0(h)

    return pre_factor * result[0] + f_im_0


def f_stp_integrand_kkr(k, w, v, gamma, h):
    pre_factor_k = np.exp(-2 * k * z_0)
    pre_factor_w = w / np.sqrt((k*v)**2 - w**2)
    im_func = np.imag(1 / e_kkr(k, w, gamma, h))
    if type(im_func) not in (int, float, np.float64):
        print('Error, type real_func = {}'.format(type(im_func)))
    integrand = pre_factor_k * pre_factor_w * im_func

    return integrand


def f_stp_kkr(v, gamma, h):
    pre_factor = 2 * (Q ** 2) / (np.pi * v)
    result = integrate.dblquad(lambda dw, dk: f_stp_integrand_kkr(dk, dw, v, gamma, h),
                               0, np.inf, lambda k: 0, lambda k: k*v, epsabs=1.49e-04, epsrel=1.49e-04)

    return pre_factor * result[0]

# Integration functions for KKR expressions (gamma = 0) of F_image and F_stopping: #####################################


def get_kappa(v):
    kappa = (2 * np.pi * n_0) / (v**2 - alpha)
    return kappa


def get_kc(h, v):
    assert(v**2 > alpha)
    kappa = get_kappa(v)

    const1 = -2 * kappa * h
    const2 = const1 * np.exp(const1) * (e_s - 1) / (e_s + 1)
    kc_1 = (-1 * const1 + spec.lambertw(const2)) / (2 * h)

    # def f(x):
    #     y = 1 - ((e_s - 1) / (e_s + 1)) * np.exp(-2 * x * h) - (x / kappa)
    #     return y
    # kc_2 = optimize.broyden1(f, 0)
    # kc_3 = optimize.newton_krylov(f, 0)

    kc = [kc_1]  # , kc_2, kc_3]
    return kc


def e_0_gamma0(k, h):
    frac = (e_s - 1)/(e_s + 1)
    denominator = 1 - frac*np.exp(-2*k*h)

    return 1 / denominator


def f_im_integrand_gamma0(k, v, h):
    kappa = get_kappa(v)

    factor_1 = k * np.exp(-2 * k * z_0)
    factor_2 = 1 / (e_0_gamma0(k, h)**2)
    factor_3 = 1 / np.sqrt(1/e_0_gamma0(k, h) + k/L)
    factor_4 = 1 / np.sqrt(1/e_0_gamma0(k, h) - k/kappa)

    integrand = factor_1 * factor_2 * factor_3 * factor_4
    return integrand


def f_im_gamma0(v, h):
    if v < np.sqrt(alpha):
        kc = np.inf

    else:
        kc_lst = get_kc(h, v)
        kc = kc_lst[0]

    pre_factor = -1 * (Q ** 2)
    result = integrate.quad(lambda dk: f_im_integrand_gamma0(dk, v, h), 0, kc, epsabs=1.49e-04, epsrel=1.49e-04)
    f_im_0 = get_f_im_0(h)

    return pre_factor * result[0] + f_im_0


def f_stp_integrand_gamma0(k, v, h):
    kappa = get_kappa(v)

    factor_1 = np.exp(-2 * k * z_0)
    factor_2 = 1 / (e_0_gamma0(k, h)**2)
    factor_3 = np.sqrt(k / (k - kappa/e_0_gamma0(k, h)))

    integrand = factor_1 * factor_2 * factor_3
    return integrand


def f_stp_gamma0(v, h):
    if v < np.sqrt(alpha):
        return 0

    else:
        kc_lst = get_kc(h, v)
        kc = kc_lst[0]

    kappa = get_kappa(v)
    pre_factor = -kappa * (Q ** 2) * np.sqrt(1 - alpha/(v**2))
    result = integrate.quad(lambda dk: f_stp_integrand_gamma0(dk, v, h), kc, np.inf,
                            epsabs=1.49e-04, epsrel=1.49e-04)

    return pre_factor * result[0]


def f_im_integrand_gamma0_hinf(k, v):
    kappa = get_kappa(v)

    factor_1 = k * np.exp(-2 * k * z_0)
    factor_3 = 1 / np.sqrt(1 + k/L)
    factor_4 = 1 / np.sqrt(1 - k/kappa)

    integrand = factor_1 * factor_3 * factor_4
    return integrand


def f_im_gamma0_hinf(v):
    if v < np.sqrt(alpha):
        kc = np.inf

    else:
        kc = get_kappa(v)

    pre_factor = -1 * (Q ** 2)
    result = integrate.quad(lambda dk: f_im_integrand_gamma0_hinf(dk, v), 0, kc, epsabs=1.49e-04, epsrel=1.49e-04)

    return pre_factor * result[0]


def f_stp_integrand_gamma0_hinf(k, v):
    kappa = get_kappa(v)

    factor_1 = np.exp(-2 * k * z_0)
    factor_3 = np.sqrt(k / (k - kappa))

    integrand = factor_1 * factor_3
    return integrand


def f_stp_gamma0_hinf(v):
    if v < np.sqrt(alpha):
        return 0

    else:
        kc = get_kappa(v)

    kappa = get_kappa(v)
    pre_factor = -kappa * (Q ** 2) * np.sqrt(1 - alpha/(v**2))
    result = integrate.quad(lambda dk: f_stp_integrand_gamma0_hinf(dk, v), kc, np.inf,
                            epsabs=1.49e-04, epsrel=1.49e-04)

    return pre_factor * result[0]


def main():
    return


if __name__ == '__main__':
    main()
