import utils.tools
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.losses import mse_loss_masked, mae_loss_masked, mse_loss_KITS, mse_loss, mae_loss, bce_logits_loss
import os.path
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from tqdm import tqdm
import logging


class Exp_Kernel_LSJSTN(Exp_Basic):
    def __init__(self, args):
        super(Exp_Kernel_LSJSTN, self).__init__(args)
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
        # if self.args.model == '...':
        #     params = []
        #     for name, module in self.model.named_modules():
        #         if name == 'alpha':
        #             params.append({'params': module.parameters(), 'lr': 0.01})
        #         elif isinstance(module, torch.nn.Module):  # 确保是一个 nn.Module 子类
        #             params.append({'params': module.parameters(), 'lr': self.args.learning_rate})
        #     model_optim = optim.Adam(params)
        # else:
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE', flag='train'):
        # special model
        if self.args.model == 'KITS' or self.args.model == 'KITS_kernel':
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
        # val_data, val_loader = self._get_data('val', self.unknow_set, self.val_set)
        test_data, test_loader = self._get_data('test', self.unknow_set, self.val_set)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss, 'train')
        # BCELogitLoss = bce_logits_loss()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            # epoch_time = time.time()
            for i, batch_data in enumerate(tqdm(train_loader, desc='Training')):
                iter_count += 1
                model_optim.zero_grad()
                gat_out, gru_out, outputs, atten_scores = self.model(x=batch_data['x'], add_data=batch_data['add_data']) # x and others
                loss = criterion(outputs, batch_data['y'], batch_data['msk']) + criterion(gat_out, batch_data['y'], batch_data['msk']) + criterion(gru_out, batch_data['y'], batch_data['msk'])
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()
            # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader)
            vali_loss = self.vali(test_data, test_loader)
            logging.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} | Kernel_Alpha: {4:.4f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, float(self.model.kernel_layer.alpha)))
            # test_loss = self.vali(test_data, test_loader, criterion)
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # print("best alpha:", self.model.alpha)
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
                gat_out, gru_out, outputs, atten_scores = self.model(x=batch_data['x'], add_data=batch_data['add_data']).detach().cpu()
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
            self.model.load_state_dict(torch.load(os.path.join(self.load_path, 'checkpoint.pth')))
        self.model.eval()
        imputations = []
        ground_truth = []
        masks = []
        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(test_loader, desc='Testing')):
                gat_out, gru_out, outputs, atten_scores = self.model(x=batch_data['x'], add_data=batch_data['add_data'])
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
        logging.info('mse:{}, rmse:{}, mae:{}, mape:{}, r2:{}, kernel_alpha:{}'.format(mse, rmse, mae, mape, r2, float(self.model.kernel_layer.alpha)))
        f = open(os.path.join(self.save_path, "result_imputation.txt"), 'a')
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        np.save(os.path.join(self.save_path, 'imputation_result.npy'), imputations)
        np.save(os.path.join(self.save_path, 'ground_truth.npy'), ground_truth)

        # logging.info('adj_sc:{}'.format(test_data.adj_sc))

        # logging.info('A_q:{}'.format(utils.tools.calculate_random_walk_matrix(test_data.adj_sc).T))

        # logging.info('adj_kernel:{}'.format(self.model.adj_matrix[0].detach().cpu().numpy()))

        # logging.info('A_q_kernel:{}'.format(self.model.A_q[0].detach().cpu().numpy()))

        logging.info('sigma:{}'.format(self.model.kernel_layer.alpha))

        return

