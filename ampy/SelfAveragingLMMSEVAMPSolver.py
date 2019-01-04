# coding=utf-8

import numpy  as np
from .utils import utils
import numpy


class SelfAveragingLMMSEVAMPSolver(object):
    """ self averaging vector approximate message passing solver (LMMSE form) """

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

        # de-noising part
        self.x_hat_1 = np.random.normal(0.0, 1.0, self.N)
        self.alpha_1 = 1.0
        self.eta_1 = 1.0
        self.gamma_2 = 1.0
        self.r_2 = np.random.normal(0.0, 1.0, self.N)

        # Linear Minimum Mean Square Error (LMMSE) Estimator part
        self.x_hat_2 = np.random.normal(0.0, 1.0, self.N)
        self.alpha_2 = 1.0
        self.eta_2 = 1.0
        self.gamma_1 = 1.0
        self.r_1 = np.random.normal(0.0, 1.0, self.N)

    def solve(self, max_iteration=50, tolerance=1e-5, message=False):
        """VAMP solver

        Args:
            max_iteration: maximum number of iterations to be used
            tolerance: stopping criterion
            message: convergence info

        Returns:
            estimated signal
        """
        convergence_flag = False
        abs_diff = 9999
        iteration_index = 9999

        for iteration_index in range(max_iteration):
            old_x_hat_1 = self.x_hat_1.copy()

            # update sequence are written here

            abs_diff = np.linalg.norm(old_x_hat_1 - self.x_hat_1) / np.sqrt(self.N)
            if abs_diff < tolerance:
                convergence_flag = True
                if message:
                    print("requirement satisfied")
                    print("abs_diff: ", abs_diff)
                    print("abs_estimate: ", np.linalg.norm(self.r))
                    print("iteration number = ", iteration_index)
                    print()
                break
        if convergence_flag:
            pass
            # print("converged")
            # print("abs_diff=", abs_diff)
            # print("estimate norm=", np.linalg.norm(self.r))
            # if np.linalg.norm(self.r) !=0.0 :
            #     print("relative diff= ", abs_diff / np.linalg.norm(self.r))
            # print("iteration num=", iteration_index)
            # print()
        else:
            print("does not converged.")
            print("abs_diff=", abs_diff)
            print("estimate norm=", np.linalg.norm(self.x_hat_1))
            if np.linalg.norm(self.x_hat_1) != 0.0:
                print("relative diff= ", abs_diff / np.linalg.norm(self.x_hat_1))
            print("iteration num=", iteration_index)
            print()

        return self.x_hat_1

    def __update_x_hat_1(self):
        """ update x_hat_1

        Returns:
            new x_hat_1
        """
        pass

    def __update_alpha_1(self):
        """update alpha_1

        Returns:
            new alpha_1
        """
        pass

    def __update_eta_1(self):
        """update eta_1

        Returns:
            new eta_1
        """
        pass

    def __update_gamma_2(self):
        """update gamma_2

        Returns:
            new gamma_2
        """
        pass

    def __update_r_2(self):
        """update r_2

        Returns:
            new r_2
        """
        pass

    def __update_x_hat_2(self):
        """update x_hat_2

        Returns:
            new x_hat_2
        """
        pass

    def __update_alpha_2(self):
        """update alpha_2

        Returns:
            new alpha_2
        """
        pass

    def __update_eta_2(self):
        """update eta_2

        Returns:
            new eta_2
        """
        pass

    def __update_gamma_1(self):
        """update gamma_1

        Returns:
            new gamma_1
        """
        pass

    def __update_r_1(self):
        """update r_1

        Returns:
            new r_1
        """
        pass
