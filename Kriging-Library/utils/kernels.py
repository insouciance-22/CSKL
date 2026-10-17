import numpy as np
from scipy.special import kn, gamma


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


def rfb_kernel_prime(A, l=1., theta=0.1):
    finite_dist = A.copy()
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = sigma ** 2 * np.exp(-(A ** 2 / (2 * l ** 2)))
    adj[adj < theta] = 0.
    return adj


def matern_kernel(A, nu=2.5, l=1., theta=0.1):
    """
    - nu: 核函数的平滑度参数, (0.5)不连续, (1.5-2.5)连续且平滑, 2.5更平滑, 1.5更局部, (∞)最平滑，接近RFB
    - l: 核函数的尺度参数, 越大节点之间关系随距离衰减变慢, 即较远的节点之间关系仍然明显, 表明对距离更不敏感, 反之，越小，越不明显
    """
    A[A == 0] = 1e-10
    factor = (np.sqrt(2 * nu) * A) / l  # 计算核函数中的 factor
    adj = (1 / (gamma(nu) * (2**(nu - 1)))) * ((2 * nu ** 0.5) * factor ** nu) * kn(nu, factor)
    adj[adj < theta] = 0.
    adj[adj == np.nan] = 0.
    return adj


def rational_quadratic_kernel(A, alpha=2., l=1., theta=0.1):
    """
    - alpha: 超参数，控制平滑度，与l同理，越大越平滑
    - l: 和matern核的l一样的作用
    """
    finite_dist = A.copy()
    finite_dist = finite_dist[~np.isinf(finite_dist)]
    sigma = finite_dist.std()
    adj = sigma ** 2 * (1 + (A ** 2) / (2 * alpha * l ** 2)) ** (-alpha)
    np.fill_diagonal(A, 1)
    adj[adj < theta] = 0.
    return adj