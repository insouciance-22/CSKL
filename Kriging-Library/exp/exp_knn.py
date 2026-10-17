from data_provider.data_factory import data_provider
from utils.metrics import metric
import os.path
import numpy as np
from tqdm import tqdm
import logging
import torch


class Exp_KNN(object):
    def __init__(self, args):
        self.unknow_set = None
        self.val_set = None
        self.args = args
        self.device = self._acquire_device()
        self.save_path = self.args.save_path

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda')
            print('Use GPU')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag, train_unknow_set=None, train_val_set=None):
        dataset, data_loader = data_provider(self.args, flag, self.device, train_unknow_set, train_val_set)
        if flag == 'train':
            self.args.test_name = dataset.test_name  # keep the test\vali load the same test_node
            self.unknow_set = dataset.unknow_set
            self.val_set = dataset.val_set
        return dataset, data_loader

    def train(self):
        _, _ = self._get_data(flag='train') # init node set
        logging.info("Unknow nodes have initialized! Mean output does not require training.")
        return

    def test(self, test=0):
        if test == 1:
            logging.info("KNN doesn't need to load any model.")
        test_data, test_loader = self._get_data('test', self.unknow_set, self.val_set)
        imputations = []
        ground_truth = []
        masks = []
        for i, batch_data in enumerate(tqdm(test_loader, desc='Testing')):
            outputs = batch_data['x']
            imputations.append(outputs.detach().cpu().numpy())
            ground_truth.append(batch_data['y'].detach().cpu().numpy())
            masks.append(batch_data['msk'].detach().cpu().numpy())
        imputations = np.concatenate(imputations, 0)
        ground_truth = np.concatenate(ground_truth, 0)
        masks = np.concatenate(masks, 0)
        logging.info('test_shape:{0}, {1}'.format(imputations.shape, ground_truth.shape))
        # save result
        mse, rmse, mae, mape, r2 = metric(imputations[masks == 1], ground_truth[masks == 1])
        logging.info('mse:{}, rmse:{}, mae:{}, mape:{}, r2:{}'.format(mse, rmse, mae, mape, r2))
        f = open(os.path.join(self.save_path, "result_imputation.txt"), 'a')
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        np.save(os.path.join(self.save_path, 'imputation_result.npy'), imputations)
        np.save(os.path.join(self.save_path, 'ground_truth.npy'), ground_truth)
        return

