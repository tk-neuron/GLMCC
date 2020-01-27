"""
define GLMCC class

usage:
> import glmcc
> glm = glmcc.GLMCC()
> glm.fit(t_sp): t_sp is relative spike time (target neuron - reference neuron)
> at, j_ij, j_ji = glm.theta[glm.m], glm.theta[-2], glm.theta[-1]

"""

import numpy as np
from scipy import special
import matplotlib.pyplot as plt


class _BaseGLM:
    """
    basic setting and functions for GLMCC
    """

    def __init__(self, bin_width, window, delay, tau, beta, theta):
        self.delta = bin_width
        self.w = window
        self.delay = delay
        self.tau = tau
        self.beta = beta

        self.m = int(2 * self.w / self.delta)
        self.k = np.arange(1, self.m + 1)  # 1 ~ 100
        self.xk = self.k * self.delta - self.w  # -49 ~ 50
        self.xk_n1 = self.xk.copy() - self.delta  # -50 ~ 49
        self.areas = ((self.xk <= - self.delay),
                      (self.xk > - self.delay) & (self.xk <= self.delay),
                      (self.xk > self.delay))

        self.theta = theta
        if self.theta is None:
            self.theta = np.zeros((self.m + 2))

    def func_f(self, s):
        """
        monosynaptic impact
        """
        fs = np.zeros(s.shape)
        fs[s >= self.delay] = np.exp((-1) * (s[s >= self.delay] - self.delay) / self.tau)
        return fs

    def log_lambda_cc(self, s):
        """
        firing probability
        """
        return self.theta[:self.m] + self.theta[-2] * self.func_f(s) + self.theta[-1] * self.func_f(-s)


class _Gk(_BaseGLM):
    """
    calculate Gk and its derivatives
    """

    def __init__(self, bin_width, window, delay, tau, beta, theta):
        super().__init__(bin_width, window, delay, tau, beta, theta)

    def gk(self):
        gk = np.zeros(self.m)
        ak_n1 = np.append(0., self.theta[:self.m][:-1])

        gk[self.areas[1]] = self.delta * np.exp(self.theta[:self.m][self.areas[1]])

        idx_approx_0 = (np.abs(self.theta[-1] * self.func_f(-self.xk)) < 1.0e-06) & self.areas[0]
        gk[idx_approx_0] = self.delta * np.exp(self.theta[:self.m][idx_approx_0])

        idx_approx_2 = (np.abs(self.theta[-2] * self.func_f(self.xk_n1)) < 1.0e-06) & self.areas[2]
        gk[idx_approx_2] = self.delta * np.exp(ak_n1[idx_approx_2])

        idx_not_approx_0 = (np.abs(self.theta[-1] * self.func_f(-self.xk)) >= 1.0e-06) & self.areas[0]
        gk[idx_not_approx_0] = self.tau * np.exp(self.theta[:self.m][idx_not_approx_0]) * (
                special.expi(self.theta[-1] * self.func_f(-self.xk[idx_not_approx_0]))
                - special.expi(self.theta[-1] * self.func_f(-self.xk_n1[idx_not_approx_0])))

        idx_not_approx_2 = (np.abs(self.theta[-2] * self.func_f(self.xk_n1)) >= 1.0e-06) & self.areas[2]
        gk[idx_not_approx_2] = self.tau * np.exp(self.theta[:self.m][idx_not_approx_2]) * (
                special.expi(self.theta[-2] * self.func_f(self.xk_n1[idx_not_approx_2]))
                - special.expi(self.theta[-2] * self.func_f(self.xk[idx_not_approx_2])))
        return gk

    def gk_first_derivative(self):
        dgk_dj_ij = np.zeros(self.m)
        dgk_dj_ji = np.zeros(self.m)

        if abs(self.theta[-2]) < 1.0e-03:
            dgk_dj_ij[self.areas[2]] = self.tau * np.exp(self.theta[:self.m][self.areas[2]]) * self.func_f(
                self.xk_n1[self.areas[2]]) * (1 - np.exp(- self.delta / self.tau))
        else:
            dgk_dj_ij[self.areas[2]] = (self.tau * np.exp(self.theta[:self.m][self.areas[2]]) / self.theta[-2]) * (
                    np.exp(self.theta[-2] * self.func_f(self.xk_n1[self.areas[2]])) -
                    np.exp(self.theta[-2] * self.func_f(self.xk[self.areas[2]])))

        if abs(self.theta[-1]) < 1.0e-03:
            dgk_dj_ji[self.areas[0]] = self.tau * np.exp(self.theta[:self.m][self.areas[0]]) * self.func_f(
                -self.xk[self.areas[0]]) * (1 - np.exp(- self.delta / self.tau))
        else:
            dgk_dj_ji[self.areas[0]] = (self.tau * np.exp(self.theta[:self.m][self.areas[0]]) / self.theta[-1]) * (
                    np.exp(self.theta[-1] * self.func_f(-self.xk[self.areas[0]])) -
                    np.exp(self.theta[-1] * self.func_f(-self.xk_n1[self.areas[0]])))

        return dgk_dj_ij, dgk_dj_ji

    def gk_second_derivative(self):
        d2gk_dj_ij2 = np.zeros(self.m)
        d2gk_dj_ji2 = np.zeros(self.m)

        if abs(self.theta[-2]) < 1.0e-3:
            d2gk_dj_ij2[self.areas[2]] = (self.tau / 2) * np.exp(self.theta[:self.m][self.areas[2]]) * self.func_f(
                self.xk_n1[self.areas[2]]) ** 2 * (1 - np.exp(-2 * self.delta / self.tau))
        else:
            hjf = self._func_h(self.theta[-2] * self.func_f(self.xk_n1[self.areas[2]])) - self._func_h(
                self.theta[-2] * self.func_f(self.xk[self.areas[2]]))
            coef = self.tau * np.exp(self.theta[:self.m][self.areas[2]]) / self.theta[-2] ** 2
            d2gk_dj_ij2 = coef * hjf

        if abs(self.theta[-1]) < 1.0e-3:
            d2gk_dj_ji2[self.areas[0]] = (self.tau / 2) * np.exp(self.theta[:self.m][self.areas[2]]) * self.func_f(
                -self.xk[self.areas[2]]) ** 2 * (1 - np.exp(-2 * self.delta / self.tau))
        else:
            hjf = self._func_h(self.theta[-1] * self.func_f(-self.xk[self.areas[0]])) - self._func_h(
                self.theta[-1] * self.func_f(-self.xk_n1[self.areas[0]]))
            coef = self.tau * np.exp(self.theta[:self.m][self.areas[0]]) / self.theta[-1] ** 2
            d2gk_dj_ji2 = coef * hjf

        return d2gk_dj_ij2, d2gk_dj_ji2

    @staticmethod
    def _func_h(x):
        return (x - 1) * np.exp(x)


class GLMCC(_Gk):
    """
    usage
    > import glmcc
    > glm = glmcc.GLMCC(bin_width, window, delay, tau, beta, theta)
    > glm.fit(t_sp)  # t_sp: np.array with shape (n_sp, 1)
    """

    def __init__(self, bin_width=1., window=50., delay=3., tau=4., beta=4000., theta=None):
        """
        :param bin_width: bin width when making cross-correlogram
        :param window: window size when making cross-correlogram
        :param delay:
        :param tau: time constant of synaptic impact
        :param beta: penalize parameter for the fluctuation of a(t)
        :param theta: model parameters (a(t), j_ij, j_ji)
        """
        super().__init__(bin_width, window, delay, tau, beta, theta)
        self.max_log_posterior = None
        self.j_thresholds = None  # statistical test for putative J_ij and J_ji

    def make_cc(self, t_sp):
        """
        make cross-correlogram from relative spike time
        """
        cc, _bins = np.histogram(t_sp, bins=self.m, range=(-self.w, self.w))
        return cc

    def fit(self, t_sp, clm=0.01, eta=0.1, max_iter=1000, j_min=-3.0, j_max=5.0):
        """
        fit the model parameters to relative spike time and CC
        """
        # initialize parameters
        t_sp = np.array(t_sp)
        cc = self.make_cc(t_sp)
        self.theta[:self.m] = np.log((1.0 + np.sum(cc)) / (2 * self.w))  # avoid zero division in log
        self.theta[[-2, -1]] = 0.1
        iter_count = 0

        while True:
            grad = self._gradient(t_sp=t_sp, cc=cc)
            hess = self._hessian()

            tmp_log_posterior = self._log_posterior(t_sp=t_sp, cc=cc)
            tmp_theta = self.theta.copy()  # save theta

            try:
                self.theta -= np.dot(np.linalg.inv(hess + clm * np.diag(np.diag(hess))), grad)
            except np.linalg.LinAlgError as e:
                print(e)
                new_log_posterior = None
                break

            # adjust J
            self.theta[-2:][self.theta[-2:] < j_min] = j_min
            self.theta[-2:][self.theta[-2:] > j_max] = j_max

            new_log_posterior = self._log_posterior(t_sp=t_sp, cc=cc)
            iter_count += 1

            if iter_count == max_iter:
                print("LM does not converge within 1000 times: breaking the loop.")
                return False

            elif new_log_posterior - tmp_log_posterior >= 0:
                if abs(new_log_posterior - tmp_log_posterior) < 1.0e-4:
                    break

                else:
                    clm *= eta
                    continue

            else:
                self.theta = tmp_theta
                clm *= (1 / eta)
                continue

        print("iterations needed: {}".format(iter_count))
        self.max_log_posterior = new_log_posterior
        self._statistical_test()
        return True

    def _statistical_test(self, z_alpha=3.29):
        """
        test whether the estimated connectivity is statistically significant
        :param z_alpha: if alpha = 0.01, set at 2.58; if alpha = 0.001, set at 3.29
        :return:
        """
        cc_0 = np.zeros(2)
        cc_0[0] = np.mean(np.exp(self.theta[int(self.w + self.delay):int(self.w + self.delay + self.tau)]))
        cc_0[1] = np.mean(np.exp(self.theta[int(self.w - self.delay - self.tau):int(self.w - self.delay)]))
        self.j_thresholds = 1.57 * z_alpha / np.sqrt(self.tau * cc_0)

    def _log_posterior(self, t_sp, cc):
        log_posterior = np.dot(cc, self.theta[:self.m]) + np.sum(self.theta[-2] * self.func_f(t_sp)) + np.sum(
            self.theta[-1] * self.func_f(-t_sp)) - np.sum(self.gk()) - (self.beta / (2 * self.delta)) * np.sum(
            (self.theta[:self.m][1:] - self.theta[:self.m][:-1]) ** 2)
        return log_posterior

    def _gradient(self, t_sp, cc):
        gradient = np.zeros(self.theta.shape)

        ak_n1 = np.append(0., self.theta[:self.m][:-1])
        ak_p1 = np.append(self.theta[:self.m][1:], 0.)
        gradient[:self.m] = cc - self.gk() + (self.beta / self.delta) * ((self._k_delta(1) - 1) * (
                self.theta[:self.m] - ak_n1) + (self._k_delta(self.m) - 1) * (self.theta[:self.m] - ak_p1))

        (dgk_dj_ij, dgk_dj_ji) = self.gk_first_derivative()

        gradient[-2] = np.sum(self.func_f(t_sp[t_sp > self.delay])) - np.sum(dgk_dj_ij)
        gradient[-1] = np.sum(self.func_f(-t_sp[t_sp < -self.delay])) - np.sum(dgk_dj_ji)

        return gradient

    def _hessian(self):
        hessian = np.zeros((self.theta.shape[0], self.theta.shape[0]))

        # first segment of hessian: m * m
        hessian[:self.m, :self.m] = (self.beta / self.delta) * (np.eye(self.m, k=-1) + np.eye(self.m, k=1))
        hessian[:self.m, :self.m][np.eye(self.m, dtype=bool)] = - self.gk() + (self.beta / self.delta) * (
                self._k_delta(1) + self._k_delta(self.m) - 2)

        # second segment of hessian: m * 2 and 2 * m (transpose)
        (dgk_dj_ij, dgk_dj_ji) = self.gk_first_derivative()
        hessian[:self.m, -2] = - dgk_dj_ij
        hessian[:self.m, -1] = - dgk_dj_ji

        hessian[[-2, -1], :self.m] = hessian[:self.m, [-2, -1]].T  # transpose

        # third segment of hessian: 2 * 2; no need to calc d2logp/dj_dj since it equals zero
        (d2gk_dj_ij2, d2gk_dj_ji2) = self.gk_second_derivative()
        hessian[-2, -2] = - np.sum(d2gk_dj_ij2)
        hessian[-1, -1] = - np.sum(d2gk_dj_ji2)
        return hessian

    def _k_delta(self, l):
        """
        kronecker's delta
        :param l: int
        :return: boolean index (where k == l)
        """
        return self.k == l

    def plot(self, t_sp, target_id, reference_id, save_path=None, save=True,
             font_size=25, font_family='Times New Roman', figure_dpi=250):
        """
        plot cross-correlogram and fitted GLM together
        """
        plt.rcParams['font.size'] = font_size
        plt.rcParams['font.family'] = font_family
        plt.rcParams['figure.dpi'] = figure_dpi
        plt.rcParams['axes.axisbelow'] = True
        plt.grid(True)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel('time [ms]')
        ax.set_ylabel('cc')
        ax.set_xticks([-self.w, 0, self.w])
        ax.set_xticklabels([-self.w, 0, self.w])

        cc = self.make_cc(np.array(t_sp))

        at = np.exp(self.theta[:self.m])
        j_ij = np.exp(self.theta[-2] * self.func_f(s=self.xk) + self.theta[:self.m])
        j_ji = np.exp(self.theta[-1] * self.func_f(s=-self.xk) + self.theta[:self.m])

        ax.bar(self.xk, cc, color='black', width=1.0)

        # connectivity undetermined: gray, positive j: magenta, negative j: cyan
        colors = {0: 'gray', 1: 'cyan', 2: 'magenta'}

        ax.plot(self.xk, j_ij, linewidth=3.0,
                color=colors[(abs(self.theta[-2]) >= self.j_thresholds[0]) * ((self.theta[-2] > 0) + 1)])
        ax.plot(self.xk, j_ji, linewidth=3.0,
                color=colors[(abs(self.theta[-1]) >= self.j_thresholds[1]) * ((self.theta[-1] > 0) + 1)])

        ax.plot(self.xk, at, linewidth=3.0, color='lime')

        ax.set_xlim(-self.w, self.w)
        ax.set_ylim(0, max(np.max(cc), np.max(j_ij), np.max(j_ji)) * 1.1)
        ax.set_title('from {} to {} (delay: {})'.format(reference_id, target_id, self.delay))

        if save:
            if save_path is None:
                print("please specify the save path")
            else:
                plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
                print("GLMCC graph saved in " + save_path)
        plt.close()

