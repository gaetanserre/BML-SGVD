import matplotlib.pyplot as plt
import torch
import seaborn as sns
from tqdm.auto import tqdm
import scipy
import numpy as np
import os

from kernel import rbf
from SVGD import svgd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_figures(x, logprob, i, fig_path):
  plt.style.use("seaborn-v0_8")
  if i == 0:
    label = "$\mu_{" + str(i) + "}$"
  else:
    label = "$\hat{\mu}_{" + str(i) + "}$"

  sns.kdeplot(x.cpu().detach().numpy(), bw_method=.4, label=label)

  x = torch.linspace(-30, 30, 1000)
  y = logprob(x.to(device)).cpu()
  plt.plot(x, torch.exp(y), "--", label="$\pi$")

  plt.xlim(-15, 15)
  plt.ylim(0, 0.6)
  plt.xlabel("$\mathcal{X}$")
  plt.legend()
  plt.savefig(fig_path + "figures/figure_" + str(i) + ".png")
  plt.savefig(fig_path + "figures/figure_" + str(i) + ".pdf")
  plt.clf()


def experiment(x, logprob, nb_iter, eta, kernel, lamb, fig_path):
  """
  x: initial particles
  logprob: log-probability function
  nb_iter: number of iterations
  eta: learning rate
  kernel: kernel function
  lamb: lambda parameter for KL bound
  fig_path: path to save figures
  """

  # Create directory to save figures
  if not os.path.isdir(fig_path):
    os.mkdir(fig_path)
    os.mkdir(fig_path + "figures/")

  KL = []
  gradients = []
  for i in tqdm(range(nb_iter)):
    ### Compute the KL divergence between pi and mu-hat_n
    nb_bins = x.shape[0]
    x_lin = torch.linspace(-30, 30, nb_bins).unsqueeze(1).to(device)

    pi_prob = torch.exp(logprob(x_lin))
    pi_prob = pi_prob / pi_prob.sum()
    pi_prob = pi_prob[:-1].squeeze(1)
    
    y = np.histogram(x.detach().cpu(), bins=x_lin.reshape((nb_bins,)).cpu())[0]
    y = y / np.sum(y)

    KL.append(scipy.stats.entropy(y, pi_prob.detach().cpu()))
    ###

    plot_figures(x, logprob, i, fig_path)

    svgd_grad = svgd(x, logprob, kernel)

    gradients.append(svgd_grad.norm().detach().cpu())

    x = x.detach() + eta * svgd_grad.detach()

  ### Plot figures
  plot_figures(x, logprob, nb_iter, fig_path)

  exp_bounds = [np.exp(-2*lamb*t) * KL[0] for t in range(len(KL))]
  plt.plot(KL, label="$KL(\pi||\hat{\mu}_t)$")
  plt.plot(exp_bounds, "--", label="$e^{-2\lambda t}KL(\pi||\mu_0)$")
  plt.xlabel("Number of iterations")
  plt.legend()
  plt.savefig(fig_path + "KL.pdf")
  np.save(fig_path + "KL.npy", np.array(KL))

  plt.clf()
  plt.plot(gradients, label="2-norm update direction")
  plt.xlabel("Number of iterations")
  plt.legend()
  plt.savefig(fig_path + "norm_direction.pdf")
  
  print(x)

  return x, KL, gradients, exp_bounds