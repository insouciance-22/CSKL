import logging
from torch.utils.data import DataLoader
from data_provider.data_loader import (Dataset_IGNNK, Dataset_KITS, Dataset_INCREASE, Dataset_IGNNK_self_A,
                                       Dataset_IGNNIK, Dataset_IGNNK_GAE, Dataset_KNN, Dataset_IGNNK_GAT,
                                       Dataset_HSTK, Dataset_IGNNK_kernel, Dataset_ST, Dataset_INCREASE_kernel,
                                       Dataset_KITS_kernel, Dataset_LSJSTN_kernel)
# from torch_geometric.loader.dataloader import DataLoader as PyGDataLoader
# from data_provider.data_loader import Dataset_KCP

data_dict = {
    'KNN': Dataset_KNN,
    'IDW': Dataset_KNN,
    'IGNNK': Dataset_IGNNK,
    'KITS': Dataset_KITS,
    'INCREASE': Dataset_INCREASE,
    'IGNNK_kernel': Dataset_IGNNK_kernel,
    'KITS_kernel': Dataset_KITS_kernel,
}


def data_provider(args, flag, device, train_unknow_set, train_val_set):
    Data = data_dict[args.model]
    shuffle_flag = False if flag == 'test' else True
    drop_last = False
    batch_size = args.batch_size
    if args.model in {'KNN', 'IDW'}:
        data_set = Data(
            k = args.k_mean,
            args = args,
            device=device,
            dataset=args.dataset,
            N_test=args.N_test,
            N_val=args.N_val,
            T_test=args.T_test,
            T_val=args.T_val,
            seq_len=args.seq_len,
            flag=flag,
            test_name=args.test_name,
            scaler_method=args.scaler_method,
            scaler_value=args.scaler_value,
            train_unknow_set=train_unknow_set,
            train_val_set=train_val_set
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
    elif args.model in {'IGNNK'}:
        data_set = Data(
            args = args,
            device=device,
            tra_s=args.tra_s,
            tra_m=args.tra_m,
            dataset=args.dataset,
            N_test=args.N_test,
            N_val=args.N_val,
            T_test=args.T_test,
            T_val=args.T_val,
            seq_len=args.seq_len,
            flag=flag,
            ignore_zero=args.ignore_zero,
            fill_zero=args.fill_zero,
            test_name=args.test_name,
            scaler_method=args.scaler_method,
            scaler_value=args.scaler_value,
            train_unknow_set=train_unknow_set,
            train_val_set=train_val_set
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
    elif args.model == 'KITS':
        data_set = Data(
            args=args,
            device=device,
            dataset=args.dataset,
            N_test=args.N_test,
            N_val=args.N_val,
            T_test=args.T_test,
            T_val=args.T_val,
            seq_len=args.seq_len,
            flag=flag,
            ignore_zero=args.ignore_zero,
            fill_zero=args.fill_zero,
            test_name=args.test_name,
            scaler_method=args.scaler_method,
            scaler_value=args.scaler_value,
            train_unknow_set=train_unknow_set,
            train_val_set=train_val_set
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=data_set.collate_fn
        )
    elif args.model == 'INCREASE':
        data_set = Data(
            args=args,
            device=device,
            k_KNN=args.k_KNN,
            interval=args.interval,
            dataset=args.dataset,
            N_test=args.N_test,
            N_val=args.N_val,
            T_test=args.T_test,
            T_val=args.T_val,
            seq_len=args.seq_len,
            flag=flag,
            ignore_zero = args.ignore_zero,
            fill_zero=args.fill_zero,
            test_name=args.test_name,
            scaler_method=args.scaler_method,
            scaler_value=args.scaler_value,
            train_unknow_set=train_unknow_set,
            train_val_set=train_val_set
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
    elif args.model == 'IGNNK_kernel':
        data_set = Data(
            args = args,
            device=device,
            tra_s=args.tra_s,
            tra_m=args.tra_m,
            dataset=args.dataset,
            N_test=args.N_test,
            N_val=args.N_val,
            T_test=args.T_test,
            T_val=args.T_val,
            seq_len=args.seq_len,
            flag=flag,
            ignore_zero=args.ignore_zero,
            fill_zero=args.fill_zero,
            test_name=args.test_name,
            scaler_method=args.scaler_method,
            scaler_value=args.scaler_value,
            train_unknow_set=train_unknow_set,
            train_val_set=train_val_set
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
    elif args.model == 'KITS_kernel':
        data_set = Data(
            args=args,
            device=device,
            dataset=args.dataset,
            N_test=args.N_test,
            N_val=args.N_val,
            T_test=args.T_test,
            T_val=args.T_val,
            seq_len=args.seq_len,
            flag=flag,
            ignore_zero=args.ignore_zero,
            fill_zero=args.fill_zero,
            test_name=args.test_name,
            scaler_method=args.scaler_method,
            scaler_value=args.scaler_value,
            train_unknow_set=train_unknow_set,
            train_val_set=train_val_set
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=data_set.collate_fn
        )
    else:
        raise ValueError("Model Data Loader No Found!")
    logging.info("{0}_len: {1}".format(flag, len(data_set)))
    return data_set, data_loader
