#
# Created in 2023 by Gaëtan Serré
#

import torch


@torch.compile
def svgd(x, logprob, kernel):
    """
    x: torch.Tensor
    logprob: callable
    kernel: callable
    """

    # Compute the gradient of the log probability
    logprob_grad = torch.autograd.functional.jacobian(logprob, x).sum(dim=0)

    # Compute the kernel matrix and its gradient
    Kxy, dxkxy = kernel(x)

    svgd_grad = (Kxy @ logprob_grad + dxkxy) / x.shape[0]

    return svgd_grad
