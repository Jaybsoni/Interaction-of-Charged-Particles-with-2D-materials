# main file for integration
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

n_0 = 0.428
alpha = np.pi * n_0
# alpha = 0
H = 4
Gamma = 0.1
e_s = 3.9
z_0 = 3
Q = 1


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

    fig2, axes2 = plt.subplots(2, 1, sharex='col')
    # Figure 2 :
    for index1, gamma in enumerate(gamma1):
        print('processing gamma {}/3'.format(index1 + 1))
        for index2, h in enumerate(h1):
            print('processing h {}/3'.format(index2 + 1))
            force_stp = []
            force_im = []

            for vx in velocity_lst:
                force_stp.append(f_stp(vx, gamma, h))
                force_im.append(f_im(vx, gamma, h))

            force_im = -1 * np.array(force_im)
            force_stp = -1 * np.array(force_stp)

            axes2[0].plot(velocity_lst, force_stp, linestyle='--', label='gamma, h = {},{}'.format(gamma, h))
            axes2[1].plot(velocity_lst, force_im, linestyle='--')

    fig2.suptitle('Figure 2 (h = 8 a.u)')
    axes2.flat[0].set(ylabel='F_stopping')
    axes2.flat[1].set(xlabel='Velocity (a.u)', ylabel='F_image')
    fig2.legend()
    plt.savefig('figure2.png')
    plt.close(fig2)

    print('Done!')


if __name__ == '__main__':
    main()
