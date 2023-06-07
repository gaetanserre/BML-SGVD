#
# Created in 2023 by Gaëtan Serré
#

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import scipy
import numpy as np
import os

from SVGD import svgd
from Adam import Adam


def plot_figures(x, pdf, i, fig_path):
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(10, 6))
    if i == 0:
        label = "$\mu_{" + str(i) + "}$"
    else:
        label = "$\hat{\mu}_{" + str(i) + "}$"

    # sns.kdeplot(x, bw_method=0.4, label=label)
    plt.hist(x, bins=x.shape[0] // 10, density=True, label=label)

    x = np.linspace(-30, 30, 1000)
    y = pdf(x)
    plt.plot(x, y, color="red", label="$\pi$")

    plt.xlim(-15, 15)
    plt.ylim(0, 0.6)
    plt.xlabel("$\mathcal{X}$")
    plt.legend()
    plt.savefig(fig_path + "figures/svgd_" + str(i) + ".png", bbox_inches="tight")
    plt.savefig(fig_path + "figures/svgd_" + str(i) + ".svg", bbox_inches="tight")
    plt.clf()


def experiment(x, pdf, dlogpdf, nb_iter, eta, kernel, lamb, fig_path):
    """
    x: initial particles
    pdf: probability function
    dlogpdf: derivative of the log probability function
    nb_iter: number of iterations
    eta: learning rate
    kernel: kernel function
    lamb: lambda parameter for KL bound
    fig_path: path to save figures
    """

    optimizer = Adam(lr=eta)

    # Create directory to save figures
    if not os.path.isdir(fig_path):
        os.mkdir(fig_path)
        os.mkdir(fig_path + "figures/")

    KL = []
    gradients = []
    plot_figures(x, pdf, 0, fig_path)
    for i in tqdm(range(nb_iter)):
        ### Compute the KL divergence between pi and mu-hat_n
        nb_bins = x.shape[0]
        x_lin = np.linspace(-30, 30, nb_bins).reshape(-1, 1)

        pi_prob = pdf(x_lin)
        pi_prob = pi_prob / pi_prob.sum()
        pi_prob = pi_prob[:-1].squeeze(1)

        y = np.histogram(x, bins=x_lin.reshape((nb_bins,)))[0]
        y = y / np.sum(y)

        KL.append(scipy.stats.entropy(y, pi_prob))
        ###

        # plot_figures(x, pdf, i, fig_path)

        svgd_grad = svgd(x, dlogpdf, kernel)

        gradients.append(np.linalg.norm(svgd_grad))

        x = optimizer.step(svgd_grad, x)  # x + eta * svgd_grad

    ### Plot figures
    plot_figures(x, pdf, nb_iter, fig_path)

    plt.figure(figsize=(10, 6))
    exp_bounds = [np.exp(-2 * lamb * t) * KL[0] for t in range(len(KL))]
    plt.plot(KL, label="$KL(\pi||\hat{\mu}_t)$")
    plt.plot(exp_bounds, "--", label="$e^{-2\lambda t}KL(\pi||\mu_0)$")
    plt.xlabel("Number of iterations")
    plt.legend()
    plt.savefig(fig_path + "KL.svg", bbox_inches="tight")
    plt.clf()

    np.save(fig_path + "KL.npy", np.array(KL))

    plt.plot(gradients, label="2-norm update direction")
    plt.xlabel("Number of iterations")
    plt.legend()
    plt.savefig(fig_path + "norm_direction.svg", bbox_inches="tight")
    plt.clf()

    print(x)

    return x, KL, gradients, exp_bounds
