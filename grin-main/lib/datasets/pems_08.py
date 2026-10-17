import os

import numpy as np
import pandas as pd

from lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils import sample_mask

from .datetime_encode import datetime_encoded


class PEMS_08(PandasDataset):
    def __init__(self, impute_zeros=False, freq='5T'):

        df, dist, mask = self.load(impute_zeros=impute_zeros)
        self.dist = dist
        u = datetime_encoded(df, 'day') # [T, Uints_num*2]
        super().__init__(dataframe=df, u=u, mask=mask, name='la', freq=freq, aggr='nearest')

    def load(self, impute_zeros=True):
        # TODO
        path = os.path.join(datasets_path['08'], 'PEMS08.npz')
        df = pd.DataFrame(np.load(path)['data'][:, :, 0]) #(17856, 170, 3)
        time_idx = pd.read_csv(os.path.join(datasets_path['08'], 'PEMS08_dates.csv'), parse_dates=['date'])['date']
        df.index = time_idx
        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='5T')
        df = df.reindex(index=date_range)
        mask = ~np.isnan(df.values)
        if impute_zeros:
            mask = mask * (df.values != 0.).astype('uint8')
            df = df.replace(to_replace=0., method='ffill')
        else:
            mask = None
        dist = self.load_distance_matrix()
        return df, dist, mask

    def load_distance_matrix(self):
        # TODO
        path = os.path.join(datasets_path['08'], 'adj_dist.npy')
        dist = np.load(path)
        return dist

    def get_similarity(self, thr=0.1, force_symmetric=False, sparse=False):
        finite_dist = self.dist.reshape(-1)
        finite_dist = finite_dist[~np.isinf(finite_dist)]
        sigma = finite_dist.std()
        adj = np.exp(-np.square(self.dist / sigma))
        adj[adj < thr] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
        return adj

    @property
    def mask(self):
        return self._mask


class MissingValuesPEMS08(PEMS_08):
    SEED = 2026

    def __init__(self, p_fault=0.0015, p_noise=0.05):
        super(MissingValuesPEMS08, self).__init__(impute_zeros=True)
        self.rng = np.random.default_rng(self.SEED)
        self.p_fault = p_fault
        self.p_noise = p_noise
        eval_mask = sample_mask(self.numpy().shape,
                                p=p_fault,
                                p_noise=p_noise,
                                min_seq=12,
                                max_seq=12 * 4,
                                rng=self.rng)
        self.eval_mask = (eval_mask & self.mask).astype('uint8')

    @property
    def training_mask(self):
        return self.mask if self.eval_mask is None else (self.mask & (1 - self.eval_mask))

    def splitter(self, dataset, val_len=0, test_len=0, window=0):
        idx = np.arange(len(dataset))
        if test_len < 1:
            test_len = int(test_len * len(idx))
        if val_len < 1:
            val_len = int(val_len * (len(idx) - test_len))
        test_start = len(idx) - test_len
        val_start = test_start - val_len
        return [idx[:val_start - window], idx[val_start:test_start - window], idx[test_start:]]