import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

n_0 = 0.428
alpha = np.pi * n_0
# alpha = 0
# H = 4
# Gamma = 0.1
e_s = 3.9
z_0 = 3
Q = 1


def get_f_im_0(h):
    const = -1 * (Q ** 2) / (4 * (z_0 + h)**2)
    const2 = (e_s - 1) / (e_s + 1)
    return const * const2


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


def chi_new(k, w, gamma):
    k_sqrd = k**2

    numerator = k_sqrd * n_0
    denominator = alpha*k_sqrd - w**2 - np.complex(0, 1)*gamma*w

    return numerator / denominator


def e_0_new(k, h):
    frac = (e_s - 1)/(e_s + 1)
    denominator = 1 - frac*np.exp(-2*k*h)

    return 1 / denominator


def e_new(k, w, gamma, h):
    return e_0_new(k, h) + chi_new(k, w, gamma) * ((2*np.pi) / k)


def f_stp_integrand(kx, ky, vx, gamma, h):
    k_sqrd = kx**2 + ky**2
    k = np.sqrt(k_sqrd)

    pre_factor = kx * np.exp(-2 * k * z_0) / k
    complex_func = np.imag(1 / e(kx, ky, vx, gamma, h))
    if type(complex_func) not in (int, float, np.float64):
        print('Error, type real_func = {}'.format(type(complex_func)))
    integrand = pre_factor * complex_func

    return integrand


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


def f_stp(vx, gamma, h):
    pre_factor = 2 * (Q ** 2) / np.pi
    result = integrate.dblquad(lambda dkx, dky: f_stp_integrand(dkx, dky, vx, gamma, h),
                               0, np.inf, lambda t: 0, lambda t: np.inf, epsabs=1.49e-04, epsrel=1.49e-04)

    return pre_factor * result[0]


def f_im_integrand_new(k, w, v, gamma, h):

    pre_factor_k = k * np.exp(-2 * k * z_0)
    pre_factor_w = 1 / np.sqrt(w**2 - (k*v)**2)
    im_func = np.imag(1 / e_new(k, w, gamma, h))
    if type(im_func) not in (int, float, np.float64):
        print('Error, type real_func = {}'.format(type(im_func)))
    integrand = pre_factor_k * pre_factor_w * im_func

    return integrand


def f_im_new(v, gamma, h):
    pre_factor = 2 * (Q ** 2) / np.pi
    result = integrate.dblquad(lambda dw, dk: f_im_integrand_new(dk, dw, v, gamma, h),
                               0, np.inf, lambda k: k*v, lambda k: np.inf, epsabs=1.49e-04, epsrel=1.49e-04)
    f_im_0 = get_f_im_0(h)

    return pre_factor * result[0] + f_im_0


def f_stp_integrand_new(k, w, v, gamma, h):
    pre_factor_k = np.exp(-2 * k * z_0)
    pre_factor_w = w / np.sqrt((k*v)**2 - w**2)
    im_func = np.imag(1 / e_new(k, w, gamma, h))
    if type(im_func) not in (int, float, np.float64):
        print('Error, type real_func = {}'.format(type(im_func)))
    integrand = pre_factor_k * pre_factor_w * im_func

    return integrand


def f_stp_new(v, gamma, h):
    pre_factor = 2 * (Q ** 2) / (np.pi * v)
    result = integrate.dblquad(lambda dw, dk: f_stp_integrand_new(dk, dw, v, gamma, h),
                               0, np.inf, lambda k: 0, lambda k: k*v, epsabs=1.49e-04, epsrel=1.49e-04)

    return pre_factor * result[0]


def main():
    velocity_lst = np.linspace(0.05, 10, num=30)
    h1 = [8]
    gamma1 = [0.1, 0.25, 0.5]

    # fig1, axes1 = plt.subplots(2, 1, sharex='col')
    # # Figure 1 :
    # force_stp = []
    # force_im = []
    #
    # for vx in velocity_lst:
    #     force_stp.append(f_stp(vx, Gamma, H))
    #     force_im.append(f_im(vx, Gamma, H))
    #
    # force_im = -1 * np.array(force_im)
    # force_stp = -1 * np.array(force_stp)
    #
    # axes1[0].plot(velocity_lst, force_stp, linestyle='--', label='h = {}'.format(h))
    # axes1[1].plot(velocity_lst, force_im, linestyle='--')
    #
    # fig1.suptitle('Figure 1 (gamma = 0.1 a.u)')
    # axes1.flat[0].set(ylabel='F_stopping')
    # axes1.flat[1].set(xlabel='Velocity (a.u)', ylabel='F_image')
    # fig1.legend()
    # plt.savefig('figure1.png')
    # plt.close(fig1)

    fig2, axes2 = plt.subplots(2, 2, sharey="row", sharex="col")
    fig3, axes3 = plt.subplots(2, 1)
    fig2.set_size_inches(23, 12)
    fig2.set_size_inches(15, 12)
    # Figure 2 :
    for index1, gamma in enumerate(gamma1):
        print('processing gamma {}/3'.format(index1 + 1))
        for index2, h in enumerate(h1):
            print('processing h {}/1'.format(index2 + 1))
            force_stp = []
            force_im = []
            force_stp_new = []
            force_im_new = []
            diff_stp = []
            diff_im = []

            for vx in velocity_lst:
                fstped = f_stp(vx, gamma, h)
                fim = f_im(vx, gamma, h)
                fstped_new = f_stp_new(vx, gamma, h)
                fim_new = f_im_new(vx, gamma, h)
                force_stp.append(fstped)
                force_im.append(fim)
                force_stp_new.append(fstped_new)
                force_im_new.append(fim_new)
                diff_stp.append(abs(fstped - fstped_new))
                diff_im.append(abs(fim - fim_new))

            force_im = -1 * np.array(force_im)
            force_stp = -1 * np.array(force_stp)
            force_im_new = -1 * np.array(force_im_new)
            force_stp_new = -1 * np.array(force_stp_new)
            diff_stp = np.array(diff_stp)
            diff_im = np.array(diff_im)

            axes2[0, 0].plot(velocity_lst, force_stp, linestyle='--', label='gamma, h = {},{}'.format(gamma, h))
            axes2[1, 0].plot(velocity_lst, force_im, linestyle='--')
            axes2[0, 1].plot(velocity_lst, force_stp_new, linestyle='--')
            axes2[1, 1].plot(velocity_lst, force_im_new, linestyle='--')
            axes3[0].plot(velocity_lst, diff_stp, linestyle='--')
            axes3[1].plot(velocity_lst, diff_im, linestyle='--')

    # fig2.title('Stopping and Image Force over velocity (h = 8 a.u)')
    axes2.flat[0].set(ylabel='F_stopping')
    axes2.flat[2].set(xlabel='Velocity (a.u)', ylabel='F_image')
    axes2.flat[1].set(ylabel='F_stopping')
    axes2.flat[3].set(xlabel='Velocity (a.u)', ylabel='F_image')
    axes3.flat[0].set(ylabel='abs diff F_stp')
    axes3.flat[1].set(xlabel='Velocity (a.u)', ylabel='abs diff F_im')
    axes2[0, 0].set_title('Integrated kx, ky as in paper')
    axes2[0, 1].set_title('Integrated k,w using K.K.Rs')
    # axes3.title('Abs difference between two methods')
    fig2.legend()
    fig2.savefig('CompareMethods_new.png')
    fig3.legend()
    fig3.savefig('CompareMetods_new_error.png')
    plt.close(fig2)
    plt.close(fig3)

    print('Done!')


if __name__ == '__main__':
    main()