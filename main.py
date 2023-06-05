#
# Created in 2023 by Gaëtan Serré
#

import sys

from kernel import rbf
from run_exp import experiment

import numpy as np
import scipy


class Mix:
    def __init__(self, means, std, pi):
        self.means = means
        self.std = std
        self.pi = pi

    def pdf(self, x):
        return np.sum(
            [
                self.pi[i] * scipy.stats.norm.pdf(x, self.means[i], self.std[i])
                for i in range(len(self.means))
            ],
            axis=0,
        ).reshape(-1, 1)

    def dlogpdf(self, x):
        C = self.pdf(x)
        D = np.sum(
            [
                self.pi[i]
                * scipy.stats.norm.pdf(x, self.means[i], self.std[i])
                * -(x - self.means[i])
                / self.std[i] ** 2
                for i in range(len(self.means))
            ],
            axis=0,
        ).reshape(-1, 1)
        return D / C


if __name__ == "__main__":
    mix = Mix([-2, 2], [1, 1], [1 / 3, 2 / 3])

    n_particles = 1000
    eta = 0.99
    kernel = rbf

    ### Experiment 1
    x = np.random.uniform(
        -10, 10, (n_particles, 1)
    )  # np.random.normal(-10, 1, (n_particles, 1))

    lamb = 0.008
    experiment(x, mix.pdf, mix.dlogpdf, int(sys.argv[1]), eta, kernel, lamb, "exp1/")

    ### Experiment 2
    x = np.random.normal(0, 0.3, (n_particles, 1))
    lamb = 0.01
    experiment(x, mix.pdf, mix.dlogpdf, int(sys.argv[1]), eta, kernel, lamb, "exp2/")
