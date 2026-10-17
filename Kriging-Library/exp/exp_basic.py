import os
import torch
from models import (IGNNK, KITS, INCREASE, IGNNK_self_A, IGNNK_GAE, IGNNK_GAT, HSTK, IGNNK_kernel, ST_model,
                    INCREASE_kernel, KITS_kernel, INCREASE_2, LSJSTN_kernel)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'IGNNK': IGNNK,
            'KITS': KITS,
            'INCREASE': INCREASE_2,
            # 'IGNNK_self_A': IGNNK_self_A,
            # 'IGNNIK': IGNNK,
            # 'IGNNK_GAE': IGNNK_GAE,
            'IGNNK_GAT': IGNNK_GAT,
            # 'HSTK': HSTK,
            'IGNNK_kernel': IGNNK_kernel,
            # 'ST': ST_model,
            # 'INCREASE_kernel': INCREASE_kernel,
            'KITS_kernel': KITS_kernel,
            # 'LSJSTN_kernel': LSJSTN_kernel
            # 'KCP': KCP
        }
        self.device = self._acquire_device()
        if args.model in self.model_dict:
            self.model = self._build_model().to(self.device)
        self.save_path = self.args.save_path
        self.load_path = self.args.load_path

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda')
            print('Use GPU')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag, unknow_set, val_set):
        pass

    def train(self):
        pass

    def vali(self, train_loader, vali_loader):
        pass

    def test(self, test):
        pass