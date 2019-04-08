# coding=utf-8

import numpy  as np
from .utils import utils
import numba


class NaiveLMMSEVAMPSolver(object):
    """ Naive VAMP Solver (diaglnal) """

    def __init__(self, A, y, regularization_strength, dumping_coefficient,
                 clip_min=1e-9, clip_max=1e9):
        self.l = regularization_strength
        self.dumping = dumping_coefficient
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.A = A.copy()
        self.y = y.copy()
        self.J = self.A.T @ self.A
        self.y_tilde = self.A.T @ self.y

        self.M, self.N = A.shape

        # message from 2 to 1
        self.r1 = np.random.normal(0.0, 1.0, self.N)
        self.q1_hat = np.ones(self.N) * 1e-2

        # variable 1 estimation
        self.x1_hat = np.random.normal(0.0, 1.0, self.N)
        self.chi1 = np.ones(self.N)  # variance
        self.eta1 = np.ones(self.N)  # precision

        # message from 1 to 2
        self.r2 = np.random.normal(0.0, 1.0, self.N)
        self.q2_hat = np.ones(self.N) * 0.1

        # variable 2 estimation
        self.x2_hat = np.random.normal(0.0, 1.0, self.N)
        self.eta2 = np.ones(self.N)  # variance
        self.chi2 = np.ones(self.N)  # precision

    def solve(self, max_iteration=50, tolerance=1e-5, message=False):
        """

        Args:
            max_iteration:
            tolerance:
            message:

        Returns:

        """
        convergence_flag = False

        for iteration_index in range(max_iteration):
            # variable 1 estimation
            h = self.r1 * self.q1_hat
            self.x1_hat = utils.update_dumping(old_x=self.x1_hat,
                                               new_x=np.heaviside(np.abs(h) - self.l, 0.5) * (
                                                       h - self.l * np.sign(h)) / self.q1_hat,
                                               dumping_coefficient=self.dumping)

            # self.chi1 = self.clip(np.heaviside(np.abs(h) - self.l, 0.5) / self.q1_hat)
            self.chi1 = utils.update_dumping(old_x=self.chi1,
                                             new_x=self.clip(np.heaviside(np.abs(h) - self.l, 0.5) / self.q1_hat),
                                             dumping_coefficient=self.dumping)

            self.eta1 = 1.0 / self.chi1

            # message from 1 to 2
            self.q2_hat = self.clip(self.eta1 - self.q1_hat)
            self.r2 = (self.eta1 * self.x1_hat - self.q1_hat * self.r1) / self.q2_hat

            # variable 2 estimation
            temp = np.linalg.inv(np.diag(self.q2_hat) + self.J)
            self.x2_hat = temp @ (self.y_tilde + self.q2_hat * self.r2)
            self.chi2 = self.clip(np.diag(temp))
            self.eta2 = 1.0 / self.chi2

            # message from 2 to 1
            # self.q1_hat = self.clip(self.eta2 - self.q2_hat)
            self.q1_hat = utils.update_dumping(old_x=self.q1_hat,
                                               new_x=self.clip(self.eta2 - self.q2_hat),
                                               dumping_coefficient=self.dumping)
            # self.r1 = (self.eta2 * self.x2_hat - self.q2_hat * self.r2) / self.q1_hat
            self.r1 = utils.update_dumping(
                old_x=self.r1,
                new_x=(self.eta2 * self.x2_hat - self.q2_hat * self.r2) / self.q1_hat,
                dumping_coefficient=self.dumping
            )

            # check convergence
            diff_x = np.linalg.norm(self.x1_hat - self.x2_hat) / np.sqrt(self.N)
            diff_chi = np.linalg.norm(self.chi1 - self.chi2) / np.sqrt(self.N)

            if max(diff_x, diff_chi) < tolerance and iteration_index > 1:
                convergence_flag = True
                break

        if convergence_flag:
            print("converged")
            print("diff x", diff_x)
            print("diff chi", diff_chi)
            print()

        else:
            print("does not converged")
            print("diff x", diff_x)
            print("diff chi", diff_chi)
            print()

    def clip(self, target):
        return np.clip(
            a=target,
            a_min=self.clip_min,
            a_max=self.clip_max
        )
