# coding=utf-8
import numpy as np

dct_matrix = None


def update_dumping(old_x, new_x, dumping_coefficient):
    return dumping_coefficient * new_x + (1.0 - dumping_coefficient) * old_x


def make_dct_matrix(n):
    """make discrete cosine matrix

    Args:
        n: size of dct matrix

    Returns:
        dct matrix with size (n,n)
    """

    n_inv = 1.0 / n

    pi = np.pi
    A = np.cos(
        [[pi * i * (2.0 * j + 1) * 0.5 * n_inv for j in range(n)] if i != 0 else [0] * n for i in range(n)]

    )
    A *= np.sqrt(2.0 / n)
    A[0] *= 1.0 / np.sqrt(2)

    return A


def make_random_dct_matrix(m, n):
    if not dct_matrix:
        B = make_dct_matrix(n)
    else:
        B = dct_matrix
    A = []
    for num in np.random.permutation(n)[:m]:
        A.append(B[num])
    A = np.array(A)
    return A


def make_gauss_matrix(m, n):
    return np.random.normal(0.0, 1.0 / np.sqrt(n), (m, n))


def make_true_parameter(n, rho):
    return np.random.normal(0.0, 1.0, n) * np.random.binomial(1, rho, n)
