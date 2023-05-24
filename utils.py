#
# Created in 2023 by Gaëtan Serré
#

import torch as th


def get_device():
    if th.cuda.is_available():
        device = th.device("cuda")
    elif th.backends.mps.is_available():
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
        device = th.device("mps")
    else:
        device = th.device("cpu")

    return device
