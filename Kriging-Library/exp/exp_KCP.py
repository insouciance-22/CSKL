from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.losses import mse_loss_masked, mae_loss_masked
import os.path
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from tqdm import tqdm
import logging


class Exp_KCP(Exp_Basic):
    def __init__(self, args):
        super(Exp_KCP, self).__init__(args)
        self.unknow_set = None
        self.val_set = None

    def _build_model(self, flag=None):
        if flag is None:
            model = self.model_dict[self.args.model].SelfGNN(self.args).float()
        else:
            model = self.model_dict[self.args.model].MLPDec(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, train_unknow_set=None, train_val_set=None):
        dataset, data_loader = data_provider(self.args, flag, self.device, train_unknow_set, train_val_set)
        if flag == 'train':
            self.args.test_name = dataset.test_name  # keep the test\vali load the same test_node
            self.unknow_set = dataset.unknow_set
            self.val_set = dataset.val_set
        return dataset, data_loader

    def _select_optimizer(self):
        if self.args.pre_training:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.self_learning_lr)
        else:
            model_optim = optim.Adam([{
                'params': self.model.model.parameters(),
                'lr': self.args.downtask_lr
                },{
                'params': self.model.up_enc.student_encoder.parameters(),
                'lr': self.args.finetune_lr
            }])
        return model_optim

    def _select_criterion(self, loss_name='MSE', flag='train'):
        if loss_name == 'MSE':
            return mse_loss_masked()
        elif loss_name == 'MAE':
            return mae_loss_masked()

    def pre_train(self):
        train_data, train_loader = self._get_data(flag='train')
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, pre_training=self.args.pre_training)
        model_optim = self._select_optimizer()
        for epoch in range(self.args.self_learning_epoch):
            iter_count = 0
            train_loss = []
            self.model.train()
            for i, batch_data in enumerate(tqdm(train_loader, desc='Training')):
                iter_count += 1
                model_optim.zero_grad()
                loss, acc = self.model(batch_data)
                loss.backward()
                model_optim.step()
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            logging.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))
            early_stopping(train_loss, self.model, self.save_path)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        return self.model

    def train(self):
        self.model = self._build_model(flag='train').to(self.device)
        if self.args.pre_training:
            self.model.up_enc.load_state_dict(torch.load(os.path.join(self.save_path, 'pre_training_checkpoint.pth')))
        else:
            self.model.up_enc.load_state_dict(torch.load(os.path.join(self.args.load_path_enc, 'pre_training_checkpoint.pth')))
        self.args.pre_training = False
        # down_stream
        train_data, train_loader = self._get_data('train', self.unknow_set, self.val_set)
        vali_data, vali_loader = self._get_data('val', self.unknow_set, self.val_set)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss, 'train')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            for i, batch_data in enumerate(tqdm(train_loader, desc='Training')):
                iter_count += 1
                model_optim.zero_grad()
                outputs = self.model(batch_data.x, batch_data) # x and others
                y = self.model.y_transfer(batch_data.y, batch_data)
                msk = self.model.y_transfer(batch_data.msk, batch_data)
                loss = criterion(outputs, y, msk)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()
            # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader)
            logging.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
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
                outputs = self.model(batch_data.x, batch_data).detach().cpu()
                ground_truth = self.model.y_transfer(batch_data.y, batch_data).detach().cpu()
                mask = self.model.y_transfer(batch_data.msk, batch_data).detach().cpu()
                # ground_truth = batch_data.y.detach().cpu()
                # mask = batch_data.msk.detach().cpu()
                loss = criterion(outputs, ground_truth, mask)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, test=0):
        test_data, test_loader = self._get_data('test', self.unknow_set, self.val_set)
        if test:
            logging.info('loading model')
            self.model = self._build_model(flag='test').to(self.device)
            self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'checkpoint.pth')))
        self.model.eval()
        imputations = []
        ground_truth = []
        masks = []
        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(test_loader, desc='Testing')):
                outputs = self.model(batch_data.x, batch_data)
                imputations.append(outputs.detach().cpu().numpy())
                ground_truth.append(self.model.y_transfer(batch_data.y, batch_data).detach().cpu())
                masks.append(self.model.y_transfer(batch_data.msk, batch_data).detach().cpu())
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

