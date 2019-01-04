# coding=utf-8
import numpy as np
from .utils import utils
import numba


class SelfAveragingAMPSolver(object):
    """ self averaging approximate message passing solver for the Standard Linear Model (SLM) """

    def __init__(self, A, y, regularization_strength, dumping_coefficient):
        """constructor

        Args:
            A: observation matrix of shape (M, N)
            y: observed value of shape (M, )
            regularization_strength: regularization parameter
            dumping_coefficient: dumping coefficient
        """
        self.A = A.copy()
        self.y = y.copy()
        self.M, self.N = A.shape
        self.alpha = self.M / self.N

        self.V = 1.0
        self.z = np.random.normal(0.0, 1.0, self.M)
        self.R = np.zeros(self.N)
        self.T = 1.0

        self.r = np.zeros(self.N)  # estimator
        self.chi = np.ones(self.N)  # variance

        self.l = regularization_strength  # regularization parameter
        self.d = dumping_coefficient  # dumping coefficient

    @numba.jit(parallel=True)
    def solve(self, max_iteration=50, tolerance=1e-5, message=False):
        """Self averaging AMP solver

        Args:
            max_iteration:
            tolerance:
            message:

        Returns:

        """
        converged = False

        for iteration_index in range(max_iteration):
            # self.V, self.z, self.R, self.T = self.__update_V(), self.__update_z(), self.__update_R(), self.__update_T()
            self.V, self.z = self.__update_V(), self.__update_z()

            self.R, self.T = self.__update_R(), self.__update_T()

            new_r, new_chi = self.__update_r(), self.__update_chi()
            old_r = self.r.copy()
            self.r = utils.update_dumping(self.r, new_r, self.d)
            self.chi = utils.update_dumping(self.chi, new_chi, self.d)

            abs_diff = np.linalg.norm(old_r - self.r) / np.sqrt(self.N)

            if abs_diff < tolerance:
                converged = True
                if message:
                    print("requirement satisfied")
                    print("abs_diff=", abs_diff)
                    print("abs_estimate=", np.linalg.norm(self.r))
                    print("iteration number=", iteration_index + 1)
                break

        if converged:
            pass
            # print("converged")
            # print("abs_diff=", abs_diff)
            # print("estimate norm=", np.linalg.norm(self.r))
            # if np.linalg.norm(self.r) != 0.0:
            #     print("relative diff= ", abs_diff / np.linalg.norm(self.r))
            # print("iteration num=", iteration_index + 1)
        else:
            print("does not converged.")
            print("abs_diff=", abs_diff)
            print("estimate norm=", np.linalg.norm(self.r))
            if np.linalg.norm(self.r) != 0.0:
                print("relative diff= ", abs_diff / np.linalg.norm(self.r))
            print("iteration num=", iteration_index + 1)
            print()

    @numba.jit(parallel=True)
    def __update_V(self):
        """update V

        Returns:
            new V
        """
        return self.chi.mean()

    @numba.jit(parallel=True)
    def __update_z(self):
        """update z

        Returns:
            new z
        """
        return self.y - self.A @ self.r + self.z * self.V / (1.0 + self.V)

    @numba.jit(parallel=True)
    def __update_R(self):
        """update R

        Returns:
            new R
        """
        return self.r + self.A.T @ self.z / self.alpha

    @numba.jit(parallel=True)
    def __update_T(self):
        """update T

        Returns:
            new T
        """
        return (1.0 + self.V) / self.alpha

    @numba.jit(parallel=True)
    def __update_r(self):
        """update r

        Returns:
            new r
        """
        return (self.R - self.l * self.T * np.sign(self.R)) * np.heaviside(np.abs(self.R) - self.l * self.T, 0.5)

    @numba.jit(parallel=True)
    def __update_chi(self):
        """update chi

        Returns:
            new chi
        """
        return self.T * np.heaviside(np.abs(self.R) - self.l * self.T, 0.5)
