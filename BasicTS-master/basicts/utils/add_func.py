import numpy as np


def rfb_kernel(A, l=1., theta=0.1, decisive_index=None):
    # the most commonly used
    finite_dist = A.copy()
    if decisive_index:
        finite_dist = finite_dist[:decisive_index, :decisive_index]
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = np.exp(-np.square(A / (l * sigma)))
    adj[adj < theta] = 0.
    return adj, sigma


