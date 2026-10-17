import copy
import datetime
import json
import os
import pathlib
from argparse import ArgumentParser

import numpy as np
import tqdm.auto
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

from lib import fillers, datasets, config
from lib.data.datamodule import SpatioTemporalDataModule
from lib.data.imputation_dataset import ImputationDataset, GraphImputationDataset
from lib.nn import models
from lib.nn.utils.metric_base import MaskedMetric
from lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from lib.utils import parser_utils, numpy_metrics, ensure_list, prediction_dataframe
from lib.utils.parser_utils import str_to_bool
from lib.nn.utils.rwr import get_position

import time

def has_graph_support(model_cls):
    return model_cls in [models.GRINet, models.MPGRUNet, models.BiMPGRUNet, models.SPINModel, models.PoGeVon]


def get_model_classes(model_str):
    if model_str == 'brits':
        model, filler = models.BRITSNet, fillers.BRITSFiller
    elif model_str == 'grin':
        model, filler = models.GRINet, fillers.GraphFiller
    elif model_str == 'mpgru':
        model, filler = models.MPGRUNet, fillers.GraphFiller
    elif model_str == 'bimpgru':
        model, filler = models.BiMPGRUNet, fillers.GraphFiller
    elif model_str == 'var':
        model, filler = models.VARImputer, fillers.Filler
    elif model_str == 'gain':
        model, filler = models.RGAINNet, fillers.RGAINFiller
    elif model_str == 'birnn':
        model, filler = models.BiRNNImputer, fillers.MultiImputationFiller
    elif model_str == 'rnn':
        model, filler = models.RNNImputer, fillers.Filler
    elif model_str == 'spin':
        model, filler = models.SPINModel, fillers.SPINFiller
    elif model_str == 'pogevon':
        model, filler = models.PoGeVon, fillers.GraphVAEFiller
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler


def get_dataset(dataset_name):
    if dataset_name[:3] == 'air':
        dataset = datasets.AirQuality(impute_nans=True, small=dataset_name[3:] == '36')
    elif dataset_name == 'bay_block':
        dataset = datasets.MissingValuesPemsBay()
    elif dataset_name == 'la_block':
        dataset = datasets.MissingValuesMetrLA() # p_fault=0.0015
    elif dataset_name == 'la_point':
        dataset = datasets.MissingValuesMetrLA(p_fault=0., p_noise=0.25)
    elif dataset_name == 'bay_point':
        dataset = datasets.MissingValuesPemsBay(p_fault=0., p_noise=0.25)
    elif dataset_name == '03_block':
        dataset = datasets.MissingValuesPEMS03()
    elif dataset_name == '03_point':
        dataset = datasets.MissingValuesPEMS03(p_fault=0., p_noise=0.25)
    elif dataset_name == '04_block':
        dataset = datasets.MissingValuesPEMS04()
    elif dataset_name == '04_point':
        dataset = datasets.MissingValuesPEMS04(p_fault=0., p_noise=0.25)
    elif dataset_name == '07_block':
        dataset = datasets.MissingValuesPEMS07()
    elif dataset_name == '07_point':
        dataset = datasets.MissingValuesPEMS07(p_fault=0., p_noise=0.25)
    elif dataset_name == '08_block':
        dataset = datasets.MissingValuesPEMS08()
    elif dataset_name == '08_point':
        dataset = datasets.MissingValuesPEMS08(p_fault=0., p_noise=0.25)
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    return dataset


def parse_args():
    # Argument parser
    # --config config/grin/bay_point.yaml --in-sample False --gpu 1
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument("--model-name", type=str, default='brits')
    parser.add_argument("--dataset-name", type=str, default='air')
    parser.add_argument("--config", type=str, default="config/bimpgru/la_block.yaml")
    # Splitting/aggregation params
    parser.add_argument('--in-sample', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)
    parser.add_argument('--aggregate-by', type=str, default='mean')
    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--scaled-target', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--grad-clip-algorithm', type=str, default='norm')
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--consistency-loss', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--whiten-prob', type=float, default=0.05)
    parser.add_argument('--pred-loss-weight', type=float, default=1.0)
    parser.add_argument('--warm-up', type=int, default=0)
    # graph params
    parser.add_argument("--adj-threshold", type=float, default=0.1)
    # gain hparams
    parser.add_argument('--alpha', type=float, default=10.)
    parser.add_argument('--hint-rate', type=float, default=0.7)
    parser.add_argument('--g-train-freq', type=int, default=1)
    parser.add_argument('--d-train-freq', type=int, default=5)

    known_args, _ = parser.parse_known_args()
    model_cls, _ = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    return args


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)
    # np.random.seed(1)

    model_cls, filler_cls = get_model_classes(args.model_name)
    dataset = get_dataset(args.dataset_name)

    ########################################
    # create logdir and save configuration #
    ########################################

    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(config['logs'], args.dataset_name, args.model_name, exp_name)
    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4, sort_keys=True)

    ########################################
    # data module                          #
    ########################################

    # instantiate dataset
    dataset_cls = GraphImputationDataset if has_graph_support(model_cls) else ImputationDataset
    if args.model_name == 'spin':
        exogenous = dict()
        exogenous['u_window'] = dataset.u.values
        d_u = dataset.u.values.shape[-1]
    else:
        exogenous = None
        d_u = None

    if args.model_name == 'pogevon':
        torch_dataset = dataset_cls(*dataset.numpy(return_idx=True),
                                    mask=dataset.training_mask,
                                    eval_mask=dataset.eval_mask,
                                    window=args.window,
                                    stride=args.stride,
                                    exogenous=exogenous,
                                    adjs=dataset.adjs,
                                    positions=dataset.positions,
                                    adj_label=dataset.adj_label)
    else:
        torch_dataset = dataset_cls(*dataset.numpy(return_idx=True),
                                    mask=dataset.training_mask,
                                    eval_mask=dataset.eval_mask,
                                    window=args.window,
                                    stride=args.stride,
                                    exogenous=exogenous)

    # get train/val/test indices
    split_conf = parser_utils.filter_function_args(args, dataset.splitter, return_dict=True)
    train_idxs, val_idxs, test_idxs = dataset.splitter(torch_dataset, **split_conf)
    # s = len(train_idxs)
    # x = int(0.5 * s)
    # start_idx = np.random.randint(0, s - x + 1)
    # train_idxs = train_idxs[start_idx: start_idx + x]

    # configure datamodule
    data_conf = parser_utils.filter_args(args, SpatioTemporalDataModule, return_dict=True)
    dm = SpatioTemporalDataModule(torch_dataset, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs,
                                  **data_conf)
    dm.setup()

    # if out of sample in air, add values removed for evaluation in train set
    if not args.in_sample and args.dataset_name[:3] == 'air':
        dm.torch_dataset.mask[dm.train_slice] |= dm.torch_dataset.eval_mask[dm.train_slice]

    # get adjacency matrix
    adj = dataset.get_similarity(thr=args.adj_threshold)
    # add: get distance
    dist = dataset.dist
    # force adj with no self loop
    np.fill_diagonal(adj, 0.)

    positions, anchors = get_position(adj)

    ########################################
    # predictor                            #
    ########################################

    # model's inputs
    if args.model_name == "pogevon":
        additional_model_hparams = dict(adj=adj, d_in=dm.d_in, n_nodes=dm.n_nodes, dist=dist, num_nodes=adj.shape[0],
                                        d_u=d_u, pos=positions, anchors=anchors)
    else:
        additional_model_hparams = dict(adj=adj, d_in=dm.d_in, n_nodes=dm.n_nodes, dist=dist, num_nodes=adj.shape[0], d_u=d_u)
    model_kwargs = parser_utils.filter_args(args={**vars(args), **additional_model_hparams},
                                            target_cls=model_cls,
                                            return_dict=True)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(F, args.loss_fn),
                           compute_on_step=True,
                           metric_kwargs={'reduction': 'none'})

    metrics = {'mae': MaskedMAE(compute_on_step=False),
               'mape': MaskedMAPE(compute_on_step=False),
               'mse': MaskedMSE(compute_on_step=False),
               'mre': MaskedMRE(compute_on_step=False)}

    # filler's inputs
    scheduler_class = CosineAnnealingLR if args.use_lr_schedule else None
    # scheduler_class = None
    additional_filler_hparams = dict(model_class=model_cls,
                                     model_kwargs=model_kwargs,
                                     optim_class=torch.optim.Adam,
                                     optim_kwargs={'lr': args.lr,
                                                   'weight_decay': args.l2_reg},
                                     loss_fn=loss_fn,
                                     metrics=metrics,
                                     scheduler_class=scheduler_class,
                                     scheduler_kwargs={
                                         'eta_min': 0.0001,
                                         'T_max': args.epochs
                                     },
                                     alpha=args.alpha,
                                     hint_rate=args.hint_rate,
                                     g_train_freq=args.g_train_freq,
                                     d_train_freq=args.d_train_freq)
    filler_kwargs = parser_utils.filter_args(args={**vars(args), **additional_filler_hparams},
                                             target_cls=filler_cls,
                                             return_dict=True)
    filler = filler_cls(**filler_kwargs)

    ########################################
    # training                             #
    ########################################

    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_mae', patience=args.patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1, monitor='val_mae', mode='min')

    logger = TensorBoardLogger(logdir, name="model")

    trainer = pl.Trainer(max_epochs=args.epochs,
                         logger=logger,
                         default_root_dir=logdir,
                         gpus=[args.gpu] if torch.cuda.is_available() else None,
                         gradient_clip_val=args.grad_clip_val,
                         gradient_clip_algorithm=args.grad_clip_algorithm,
                         callbacks=[early_stop_callback, checkpoint_callback])
    start = time.time()
    trainer.fit(filler, datamodule=dm)
    end = time.time()
    train_time_seconds = end - start
    print(f"\n[Direct Timing] Total training time: {train_time_seconds:.2f} seconds")



    ########################################
    # testing                              #
    ########################################

    filler.load_state_dict(torch.load(checkpoint_callback.best_model_path,
                                      lambda storage, loc: storage)['state_dict'])
    filler.freeze()
    trainer.test()
    filler.eval()

    if torch.cuda.is_available():
        filler.cuda()

    start1 = time.time()

    with torch.no_grad():
        y_true, y_hat, mask = filler.predict_loader(dm.test_dataloader(), return_mask=True)

    end1 = time.time()

    y_hat = y_hat.detach().cpu().numpy().reshape(y_hat.shape[:3])  # reshape to (eventually) squeeze node channels

    # Test imputations in whole series
    eval_mask = dataset.eval_mask[dm.test_slice]
    df_true = dataset.df.iloc[dm.test_slice]
    metrics = {
        'mae': numpy_metrics.masked_mae,
        'rmse': numpy_metrics.masked_rmse,
        'mse': numpy_metrics.masked_mse,
        'mre': numpy_metrics.masked_mre,
        'mape': numpy_metrics.masked_mape
    }
    # Aggregate predictions in dataframes+
    index = dm.torch_dataset.data_timestamps(dm.testset.indices, flatten=False)['horizon']
    aggr_methods = ensure_list(args.aggregate_by)
    df_hats = prediction_dataframe(y_hat, index, dataset.df.columns, aggregate_by=aggr_methods)
    df_hats = dict(zip(aggr_methods, df_hats))
    results = {}
    for aggr_by, df_hat in df_hats.items():
        # Compute error
        results[aggr_by] = {}
        print(f'- AGGREGATE BY {aggr_by.upper()}')
        for metric_name, metric_fn in metrics.items():
            error = metric_fn(df_hat.values, df_true.values, eval_mask).item()
            print(f' {metric_name}: {error:.4f}')
            results[aggr_by][metric_name] = float(error)
    if hasattr(filler.model, 'kernel_layer'):
        kernel_layer = filler.model.kernel_layer
        results['kernel_type'] = filler.model.kernel_type
        results['kernel_hparams'] = {
            'kernel_alpha': float(kernel_layer.kernel_alpha),
            'kernel_theta': float(kernel_layer.theta),
            'kernel_use_alpha': bool(kernel_layer.use_alpha),
            'kernel_no_self_loop': bool(kernel_layer.no_self_loop)
        }

    results['train_time_seconds'] = float(train_time_seconds)
    results['epochs_completed'] = int(trainer.current_epoch + 1)
    results['global_step'] = int(trainer.global_step)
    results['train_time_per_step_seconds'] = (
        float(train_time_seconds) / max(int(trainer.global_step), 1)
    )

    with open(os.path.join(logdir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    if filler.model.kernel_layer.use_alpha:
        if hasattr(filler.model.kernel_layer, "alpha3"):
            print("复合α模型:")
            print(
                "alpha1, mean:{:.2f}, max:{:.2f}, min:{:.2f}".format(float(torch.mean(filler.model.kernel_layer.alpha)),
                                                                    float(torch.max(filler.model.kernel_layer.alpha)),
                                                                    float(torch.min(filler.model.kernel_layer.alpha))))
            print(
                "alpha2, mean:{:.2f}, max:{:.2f}, min:{:.2f}".format(float(torch.mean(filler.model.kernel_layer.alpha2)),
                                                                    float(torch.max(filler.model.kernel_layer.alpha2)),
                                                                    float(torch.min(filler.model.kernel_layer.alpha2))))
            print(
                "alpha3, mean:{:.2f}, max:{:.2f}, min:{:.2f}".format(float(torch.mean(filler.model.kernel_layer.alpha3)),
                                                                    float(torch.max(filler.model.kernel_layer.alpha3)),
                                                                    float(torch.min(filler.model.kernel_layer.alpha3))))
            flag_file = os.path.join(logdir, f"model_with_multy_alphas.flag")
            a1 = filler.model.kernel_layer.alpha.detach().cpu().numpy()
            a2 = filler.model.kernel_layer.alpha2.detach().cpu().numpy()
            a3 = filler.model.kernel_layer.alpha3.detach().cpu().numpy()
            np.save(os.path.join(logdir, 'alpha1.npy'), a1)
            np.save(os.path.join(logdir, 'alpha2.npy'), a2)
            np.save(os.path.join(logdir, 'alpha3.npy'), a3)
            if hasattr(filler.model.kernel_layer, "w_1"):
                w1 = filler.model.kernel_layer.w_1.detach().cpu().numpy()
                w2 = filler.model.kernel_layer.w_2.detach().cpu().numpy()
                w3 = filler.model.kernel_layer.w_3.detach().cpu().numpy()
                np.save(os.path.join(logdir, 'w1.npy'), w1)
                np.save(os.path.join(logdir, 'w2.npy'), w2)
                np.save(os.path.join(logdir, 'w3.npy'), w3)
            if hasattr(filler.model.kernel_layer, "logits"):
                pi = F.softmax(filler.model.kernel_layer.logits, dim=-1).detach().cpu().numpy()
                np.save(os.path.join(logdir, 'pi.npy'), pi)
            open(flag_file, "w").close()
        else:
            print("单α模型:")
            print(
                "alpha, mean:{:.2f}, max:{:.2f}, min:{:.2f}".format(float(torch.mean(filler.model.kernel_layer.alpha)),
                                                                   float(torch.max(filler.model.kernel_layer.alpha)),
                                                                   float(torch.min(filler.model.kernel_layer.alpha))))
            print(filler.model.kernel_layer.alpha)
            print("theta:",filler.model.kernel_layer.theta)
            print("init_alpha:", filler.model.kernel_layer.kernel_alpha)
            a = filler.model.kernel_layer.alpha.detach().cpu().numpy()
            # np.save(os.path.join(logdir, 'alpha.npy'), a)
            flag_file = os.path.join(logdir, f"model_with_alpha.flag")
            open(flag_file, "w").close()
        np.save(os.path.join(logdir, 'dist.npy'), dist)
    else:
        print("use static alpha 1")
        flag_file = os.path.join(logdir, f"model_without_alpha.flag")
        open(flag_file, "w").close()
    if hasattr(filler.model, "adj_module"):
        filler.model.eval()
        with torch.no_grad():
            A_adp = filler.model.adj_module()
        np.save(os.path.join(logdir, 'A_adp.npy'), A_adp.cpu().numpy())

    mask = mask.detach().cpu().numpy().reshape(mask.shape[:3])
    y_true = y_true.detach().cpu().numpy().reshape(y_true.shape[:3])

    # np.save(os.path.join(logdir, 'imputation_result.npy'), y_hat)
    # np.save(os.path.join(logdir, 'ground_truth.npy'), y_true)
    # np.save(os.path.join(logdir, 'mask.npy'), mask)
    print(logdir)

    print(f"\n[Direct Timing] Total training time: {end - start:.2f} seconds")
    print(f"\n[Direct Timing] Total testing time: {end1 - start1:.2f} seconds")

    # return y_true, y_hat, mask
    return results["mean"]


if __name__ == '__main__':
    # args = parse_args()
    # print(args.config)
    # run_experiment(args)
    # print(args.config)
    args = parse_args()
    all_results = []
    if args.seed == -1:
        seeds = [2020, 2021, 2022]
    else:
        seeds = [args.seed]
    for seed in seeds:
        args.seed = seed
        print(f"Running experiment with seed = {seed}")
        res = run_experiment(args)
        all_results.append(res)

    final_results = {}
    for metric in all_results[0].keys():
        values = np.array([r[metric] for r in all_results])
        final_results[metric] = {
            "mean": values.mean(),
            "std": values.std(ddof=0)
        }
    print("Final results over multiple seeds:")
    for k, v in final_results.items():
        print(f"{k}: {v['mean']:.3f} ± {v['std']:.3f}")
