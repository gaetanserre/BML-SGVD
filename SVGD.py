#
# Created in 2023 by Gaëtan Serré
#


def svgd(x, dlogpdf, kernel):
    """
    x: np.array
    dlogpdf: callable
    kernel: callable
    """

    # Compute the gradient of the log probability
    logpdf_grad = dlogpdf(x)

    # Compute the kernel matrix and its gradient
    Kxy, dxkxy = kernel(x)

    svgd_grad = (Kxy @ logpdf_grad + dxkxy) / x.shape[0]

    return svgd_grad
