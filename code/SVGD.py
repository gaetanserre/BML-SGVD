import torch

@torch.compile
def svgd(x, logprob, kernel):
  """
  x: torch.Tensor
  logprob: callable
  kernel: callable
  """

  x_ = x.clone()

  # Compute the gradient of the log probability
  logprob_grad = torch.autograd.functional.jacobian(logprob, x).sum(dim=0)

  # Compute the kernel matrix
  k = kernel(x, x_)

  # Compute the divergence of the kernel
  j = torch.autograd.functional.jacobian(lambda y: kernel(y, x_), x).sum(dim=0).sum(dim=0)

  svgd_grad = (k @ logprob_grad + j) / x.shape[0]

  return svgd_grad