import torch
from tqdm.auto import tqdm
import sys

from kernel import rbf
from exp import experiment

if __name__ == "__main__":

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  w = torch.distributions.Categorical(torch.tensor([1/5, 4/5], device=device))
  means = torch.tensor([-5, 5]).to(device)
  std = torch.tensor([1, 1]).to(device)
  mix = torch.distributions.MixtureSameFamily(w, torch.distributions.Normal(means, std))

  n_particles = 100
  eta = 0.06
  kernel = rbf

  ### Experiment 1
  """ x = torch.distributions.Normal(-10, 1).sample((n_particles,)).to(device)
  lamb = 0.0008
  experiment(x, mix.log_prob, int(sys.argv[1]), eta, kernel, lamb) """

  ### Experiment 2
  x = torch.distributions.Normal(0, 0.3).sample((n_particles,)).to(device)
  lamb = 0.001
  experiment(x, mix.log_prob, int(sys.argv[1]), eta, kernel, lamb)