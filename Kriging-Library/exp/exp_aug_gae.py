from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.losses import mse_loss_masked, mae_loss_masked, mse_loss_KITS
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from tqdm import tqdm
import logging


class Exp_GAE(Exp_Basic):
    def __init__(self, args):
        super(Exp_GAE, self).__init__(args)
        self.unknow_set = None
        self.val_set = None

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        # if self.args.use_multi_gpu and self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, train_unknow_set=None, train_val_set=None):
        dataset, data_loader = data_provider(self.args, flag, self.device, train_unknow_set, train_val_set)
        if flag == 'train':
            self.args.test_name = dataset.test_name  # keep the test\vali load the same test_node
            self.unknow_set = dataset.unknow_set
            self.val_set = dataset.val_set
        return dataset, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE', flag='train'):
        # special model
        if self.args.model == 'KITS':
            if flag == 'train':
                return mse_loss_KITS()
            else:
                return mse_loss_masked()
        # only calculate  loss of the unknow_set(masked) or full_set
        if loss_name == 'MSE':
            return mse_loss_masked()
        elif loss_name == 'MAE':
            return mae_loss_masked()

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data('val', self.unknow_set, self.val_set)
        test_data, test_loader = self._get_data('test', self.unknow_set, self.val_set)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss, 'train')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            # epoch_time = time.time()
            for i, batch_data in enumerate(tqdm(train_loader, desc='Training')):
                iter_count += 1
                model_optim.zero_grad()
                outputs, adj_new = self.model(x=batch_data['x'], add_data=batch_data['add_data']) # x and others
                adj_t = batch_data['add_data']['A']
                # norm_w = adj_t.shape[0] ** 2 / float((adj_t.shape[0] ** 2 - adj_t.sum()) * 2)
                loss = criterion(outputs, batch_data['y'], batch_data['msk'])
                ep_loss = F.mse_loss(adj_t, adj_new)
                loss += 0.5 * ep_loss
                train_loss.append(loss.item())

                # if (i + 1) % 100 == 0:
                #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()

                loss.backward()
                model_optim.step()

            # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader)
            vali_loss = self.vali(test_data, test_loader)
            logging.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            # test_loss = self.vali(test_data, test_loader, criterion)
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, self.save_path)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(self.save_path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader):
        total_loss = []
        criterion = self._select_criterion(self.args.loss, 'val')
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(vali_loader, desc='Validating')):
                outputs, _ = self.model(x=batch_data['x'], add_data=batch_data['add_data'])
                outputs = outputs.detach().cpu()
                ground_truth = batch_data['y'].detach().cpu()
                mask = batch_data['msk'].detach().cpu()
                loss = criterion(outputs, ground_truth, mask)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, test=0):
        test_data, test_loader = self._get_data('test', self.unknow_set, self.val_set)
        if test:
            logging.info('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'checkpoint.pth')))
        self.model.eval()
        imputations = []
        ground_truth = []
        masks = []
        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(test_loader, desc='Testing')):
                outputs, _ = self.model(x=batch_data['x'], add_data=batch_data['add_data'])
                imputations.append(outputs.detach().cpu().numpy())
                ground_truth.append(batch_data['y'].detach().cpu().numpy())
                masks.append(batch_data['msk'].detach().cpu().numpy())
        imputations = np.concatenate(imputations, 0)
        ground_truth = np.concatenate(ground_truth, 0)
        masks = np.concatenate(masks, 0)
        imputations = test_data.inverse_transform(imputations)
        ground_truth = test_data.inverse_transform(ground_truth)
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

