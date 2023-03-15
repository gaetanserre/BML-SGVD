import torch

@torch.compile
def rbf(x, y):
  """
  x: torch.Tensor
  y: torch.Tensor
  """
  pairwise_dists = torch.abs(x[:, None] - y[None, :])
  quad = (pairwise_dists / 0.2) ** 2
  return torch.exp(-0.5 * quad)