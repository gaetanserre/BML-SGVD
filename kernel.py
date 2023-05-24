#
# Created in 2023 by Gaëtan Serré
#

import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from utils import get_device


def rbf(x, h=-1):
    x = x.cpu().detach().numpy().reshape(-1, 1)
    sq_dist = pdist(x)
    pairwise_dists = squareform(sq_dist) ** 2
    if h < 0:  # if h < 0, using median trick
        h = np.median(pairwise_dists)
        h = np.sqrt(0.5 * h / np.log(x.shape[0] + 1))

    # compute the rbf kernel
    Kxy = np.exp(-pairwise_dists / h**2 / 2)

    dxkxy = (x * Kxy.sum(axis=1).reshape(-1, 1) - Kxy @ x).reshape(
        x.shape[0], x.shape[1]
    ) / (h**2)

    device = get_device()
    Kxy = Kxy.astype(np.float32)
    dxkxy = dxkxy.astype(np.float32).reshape(-1)
    return torch.from_numpy(Kxy).to(device), torch.from_numpy(dxkxy).to(device)
