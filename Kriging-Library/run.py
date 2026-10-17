import argparse
import os
import torch
from exp.exp_normal import Exp_Normal
from exp.exp_kernel import Exp_Kernel
# from exp.exp_try import Exp_Try
# from exp.exp_KCP import Exp_KCP
# from exp.exp_aug_gae import Exp_GAE
# from exp.exp_knn import Exp_KNN
# from exp.exp_kernel_LSJSTN import Exp_Kernel_LSJSTN
from utils.print_args import log_args
from utils.tools import count_parameters
import random
import numpy as np
from datetime import datetime
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kriging')

    # basic config
    parser.add_argument('--seed', default=2020, type=int, help='fix seed')
    parser.add_argument('--test_name', type=str, default='', help='file_name of test_node and vali_node,like seed2022_30%.npz')
    parser.add_argument('--model', type=str, default='IGNNK_kernel', help='model name, options: [IGNNK, KITS, INCREASE, IGNNIK], [IGNNK_self_A,], [KCP,]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    # kernel
    parser.add_argument('--use_alpha', default=1, type=int)
    parser.add_argument('--kernel_alpha', default=1., type=float)
    parser.add_argument('--use_attn', default=0, type=int)

    # data loader
    parser.add_argument('--dataset', type=str, default='metr_la', help='dataset type, options: [metr_la, pems_bay,PEMS03]')
    parser.add_argument('--scaler_value', default=100, type=int, help='only for IGNNK, options:[100, 80]')
    parser.add_argument('--T_test', default=0.3, type=float, help='train_seq_rate of all')
    parser.add_argument('--T_val', default=0, type=float, help='val_seq_rate of train')
    # parser.add_argument('--scaler_method', default='standard', type=str, help='normalization, options: [fixed, standard]')

    # kriging config
    parser.add_argument('--N_test', default=0.3, type=float, help='test_node of all, missing_rate')
    parser.add_argument('--N_val', default=0, type=float, help='val_node of train')
    parser.add_argument('--seq_len', default=16, type=int, help='kriging sequence length')
    parser.add_argument('--ignore_zero', default=1, type=bool, help='ignore zero idx of source data')
    parser.add_argument('--fill_zero', default='knn', type=str, help='impute zero idx with ffill, options: [knn(10), ffill]')

    # model define
    # (KNN, IDW)
    parser.add_argument('--k_mean', default=4, type=int, help='mean of k-top nearest neighbors ')
    # (IGNNK, IGNNIK, KITS, GAT)
    parser.add_argument('--d', default=100, type=int, help='hidden_dim')
    parser.add_argument('--ST_d', default=32, type=int, help='hidden_dim')
    # (IGNNK, IGNNIK)
    parser.add_argument('--k_DGCN', default=1, type=int, help='for DGCN')
    parser.add_argument('--tra_s', default=0.8, type=float, help='train_nodes of all in every batch')
    parser.add_argument('--tra_m', default=0.3, type=float, help='masked as unknow_nodes of tra_s')
    # (KITS)
    parser.add_argument('--d_in', default=1, type=int, help='num features')
    parser.add_argument('--d_kits', default=32, type=int, help='num features')
    parser.add_argument('--use_adj_drop', default=0, type=int, help='dropout')
    parser.add_argument('--use_init', default=0, type=int, help='model param init')
    # (INCREASE)
    parser.add_argument('--k_KNN', default=15, type=int, help='keep k-top nearest neighbors of every node')
    parser.add_argument('--d_linear', default=64, type=int, help='dim of linear in INCREASE')
    parser.add_argument('--use_backward', default=1, type=int, help='create and use backward data')
    # (IGNNK_self_A)
    parser.add_argument('--num_nodes_tra', default=150, type=int, help='the size of self_A, options: []')
    # (IGNNK_GAE)
    parser.add_argument('--dim_h', default=128, type=int, help='')
    parser.add_argument('--dim_z', default=32, type=int, help='')
    parser.add_argument('--gae', default=0, type=int, help='')
    # (IGNNK_GAT)
    parser.add_argument('--dropout', default=0.6, type=int, help='')
    parser.add_argument('--alpha_gat', default=0.2, type=int, help='0.01-0.2')
    parser.add_argument('--heads_gat', default=8, type=int, help='')
    # (HSTK)
    parser.add_argument('--L_HSTK', default=100, type=int, help='')
    # (KCP)
    parser.add_argument('--emb_size', default=64, type=int, help='嵌入向量维度')
    parser.add_argument('--dec_hid', default=32, type=int, help='解码器隐藏维度')
    parser.add_argument('--nmb_p', default=5, type=int, help='原型头中原型的数量')
    parser.add_argument('--tra_aug', default=0.2, type=float, help='增强节点的比例')
    parser.add_argument('--rep_num', default=2, type=int, help='数据量增强扩充倍数')
    parser.add_argument('--self_learning_epoch', default=100, type=int, help='自学习epoch')
    parser.add_argument('--self_learning_lr', default=0.001, type=float, help='自学习学习率')
    parser.add_argument('--downtask_hid', default=64, type=int, help='下游解码层隐藏维度')
    parser.add_argument('--downtask_lr', default=0.005, type=float, help='下游解码学习率')
    parser.add_argument('--finetune_lr', default=0.0001, type=float, help='下游微调学习率')
    parser.add_argument('--pre_training', default=0, type=int, help='是否需要预训练')
    parser.add_argument('--load_path', default='', type=str, help='读取预训练模型')
    # (LSJSTN)
    parser.add_argument('--d_lsjstn', default=32, type=int, help='')

    # optimization
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--lradj', type=str, default='type0', help='adjust learning rate', )

    # Device
    parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
    # parser.add_argument('--gpu', type=int, default=2, help='gpu')
    # parser.add_argument('--use_multi_gpu', type=int, default=0, help='use multiple gpus')
    # parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    if args.seed != -1:
        fix_seed = args.seed
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

    # interval of dataset
    if args.dataset == 'metr_la':
        args.interval = 5
        args.num_node = 207
    elif args.dataset == 'pems_bay':
        args.interval = 5
        args.num_node = 325
    elif args.dataset in {'AQI', 'AQI-36'}:
        args.interval = 60
        args.num_node = 437
    elif args.dataset == 'PEMS03':
        args.interval = 5
        args.num_node = 358
    elif args.dataset == 'PEMS04':
        args.interval = 5
        args.num_node = 307
    elif args.dataset == 'PEMS07':
        args.interval = 5
        args.num_node = 883
    elif args.dataset == 'PEMS08':
        args.interval = 5
        args.num_node = 170

    if args.model == 'ST':
        minutes_per_day = 24 * 60
        args.steps_per_day = int(minutes_per_day / args.interval)

    if args.model in {'IGNNK', 'IGNNK_kernel'}:
        args.scaler_method = 'fixed'
    else:
        args.scaler_method = 'standard'

    # save path
    args.time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.save_path = os.path.join('logs', args.dataset, args.model, args.time_stamp)  # name/model/time_stamp/file_name
    args.load_path = os.path.join('logs', args.dataset, args.model, args.load_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # log
    log_file = os.path.join(args.save_path, 'log.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    # print_args(args)
    log_args(args)

    if args.model in {'IGNNK', 'INCREASE', 'KITS', 'IGNNIK', 'IGNNK_GAT', 'HSTK', 'ST'}:
        Exp = Exp_Normal
    elif args.model in {'IGNNK_kernel', 'INCREASE_kernel', 'KITS_kernel'}:
        Exp = Exp_Kernel
    # elif args.model in {'IGNNK_self_A'}:
    #     Exp = Exp_Try
    # elif args.model in {'KCP'}:
    #     Exp = Exp_KCP
    #     args.load_path_enc = os.path.join('logs', args.dataset, args.model, args.load_path)
    # elif args.model in {'IGNNK_GAE'}:
    #     Exp = Exp_GAE
    # elif args.model in {'KNN', 'IDW'}:
    #     Exp = Exp_KNN
    # elif args.model == 'LSJSTN_kernel':
    #     Exp = Exp_Kernel_LSJSTN
    else:
        raise ValueError("Model Name No Found!")
        # args.pre_training = True

    if args.is_training:
        for ii in range(args.itr):
            if args.pre_training:
                exp = Exp(args)
                logging.info('>>>>>>>start pre_training<<<<<<<<<<<<<<<<<<<<<<<<<<')
                exp.pre_train()
            else:
                exp = Exp(args)  # set experiments
            total_params, trainable_params = count_parameters(exp.model)
            logging.info("Total params:{0}, trainable params:{1}".format(total_params, trainable_params))
            logging.info('>>>>>>>start training<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.train()

            logging.info('>>>>>>>start testing<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test()
            torch.cuda.empty_cache()
    else:
        ii = 0
        exp = Exp(args)  # set experiments
        logging.info('>>>>>>>start testing<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(test=1)
        torch.cuda.empty_cache()