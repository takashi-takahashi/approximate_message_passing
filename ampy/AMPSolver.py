# coding=utf-8

import numpy as np
from .utils import utils
import numba


class AMPSolver(object):
    """ approximate message passing solver for the Standard Linear Model (SLM) """

    def __init__(self, A, y, regularization_strength, dumping_coefficient):
        """constructor

        Args:
            A: observation matrix of shape (M, N)
            y: observed value of shape (M, )
            regularization_strength: regularization parameter
            dumping_coefficient: dumping coefficient
        """
        self.A = A.copy()
        self.A2 = self.A * self.A  # A squared

        self.y = y.copy()
        self.M, self.N = A.shape

        self.z = np.random.normal(0.0, 1.0, self.M)
        self.V = np.random.uniform(0.5, 1.0, self.M)
        self.R = np.random.normal(0.0, 1.0, self.N)
        self.T = np.random.uniform(0.5, 1.0, self.N)

        self.r = np.zeros(self.N)  # estimator
        self.chi = np.ones(self.N)  # variance

        self.l = regularization_strength  # regularization parameter
        self.d = dumping_coefficient  # dumping coefficient

    @numba.jit(parallel=True)
    def solve(self, max_iteration=50, tolerance=1e-5, message=False):
        """AMP solver

        Args:
            max_iteration: maximum number of iterations to be used
            tolerance: stopping criterion
            message: convergence info

        Returns:
            estimated signal
        """
        convergence_flag = False
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
                convergence_flag = True
                if message:
                    print("requirement satisfied")
                    print("abs_diff: ", abs_diff)
                    print("abs_estimate: ", np.linalg.norm(self.r))
                    print("iteration number = ", iteration_index + 1)
                    print()
                break
        if convergence_flag:
            pass
            # print("converged")
            # print("abs_diff=", abs_diff)
            # print("estimate norm=", np.linalg.norm(self.r))
            # if np.linalg.norm(self.r) !=0.0 :
            #     print("relative diff= ", abs_diff / np.linalg.norm(self.r))
            # print("iteration num=", iteration_index + 1)
            # print()
        else:
            print("does not converged.")
            print("abs_diff=", abs_diff)
            print("estimate norm=", np.linalg.norm(self.r))
            if np.linalg.norm(self.r) !=0.0 :
                print("relative diff= ", abs_diff / np.linalg.norm(self.r))
            print("iteration num=", iteration_index + 1)
            print()

    @numba.jit(parallel=True)
    def __update_V(self):
        """ update V

        Returns:
            new V
        """
        return self.A2 @ self.chi

    @numba.jit(parallel=True)
    def __update_z(self):
        """ update z

        Returns:
            new z
        """
        return self.y - self.A @ self.r + (self.V / (1.0 + self.V)) * self.z

    @numba.jit(parallel=True)
    def __update_R(self):
        """ update R

        Returns:
            new R
        """
        v1 = self.A.T @ (self.z / (1.0 + self.V))
        v2 = self.A2.T @ (1.0 / (1.0 + self.V))
        return self.r + v1 / v2

    @numba.jit(parallel=True)
    def __update_T(self):
        """ update T

        Returns:
            new T
        """
        v = self.A2.T @ (1.0 / (1.0 + self.V))
        return 1.0 / v

    @numba.jit(parallel=True)
    def __update_r(self):
        """ update r

        Returns:
            new r
        """
        return (self.R - self.l * self.T * np.sign(self.R)) * np.heaviside(np.abs(self.R) - self.l * self.T, 0.5)

    @numba.jit(parallel=True)
    def __update_chi(self):
        """ update chi

        Returns:
            new chi
        """
        return self.T * np.heaviside(np.abs(self.R) - self.l * self.T, 0.5)

    def show_me(self):
        """ debug method """
        pass
