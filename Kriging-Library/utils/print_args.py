import logging


def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"kriging Type:":<20}{args.kriging_type:<20}{"Is Training:":<20}{args.is_training:<20}')
    print(f'  {"Model:":<20}{args.model:<20}')

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}')
    print(f'  {"Dataset:":<20}{args.dataset:<20}')
    print(f'  {"Test File:":<20}{args.test_name:<20}')
    print(f'  {"Temporal Split : Test Rate:":<20}{args.T_test:<20}{"Vali Rate:":<20}{args.T_val:<20}')
    print(f'  {"Scaler Method:":<20}{args.scaler_method:<20}')
    if args.scaler_method == 'fixed':
        print(f'  {"Scaler Value:":<20}{args.scaler_value:<20}')

    print("\033[1m" + "Kriging Config" + "\033[0m")
    print(f'  {"Spatio Split : Test Rate:":<20}{args.N_test:<20}{"Vali Rate:":<20}{args.N_val:<20}')
    print(f'  {"Sequence Length:":<20}{args.seq_len:<20}')
    print(f'  {"Ignore Zero:":<20}{args.ignore_zero:<20}')
    if args.ignore_zero:
        print(f'  {"Impute Zero:":<20}{args.impute_zero:<20}')

    print("\033[1m" + "Model Define" + "\033[0m")
    print(f'  {"Hidden Dim:":<20}{args.d:<20}')
    if args.model == 'IGNNK':
        print(f'  {"K For DGCN:":<20}{args.K:<20}')
        print(f'  {"Sample Rate For Training:":<20}{args.tra_s:<20}')
        print(f'  {"Unknow Rate For Sample:":<20}{args.tra_m:<20}')

    print("\033[1m" + "Optimization" + "\033[0m")
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}')
    print(f'  {"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Patience:":<20}{args.patience:<20}')
    print(f'  {"Learning Rate:":<20}{args.learning_rate:<20}')
    print(f'  {"Loss Function:":<20}{args.loss:<20}{"Loss Masked:":<20}{args.loss_masked:<20}')
    print(f'  {"Learning Rate Adjust:":<20}{args.lradj:<20}')

    print("\033[1m" + "Device" + "\033[0m")
    if args.use_gpu:
        print(f'  {"Use GPU:":<20}{args.use_gpu:<20}')
        print(f'  {"GPU ID:":<20}{args.gpu:<20}')
    elif args.use_multi_gpu:
        print(f'  {"Use Multi-GPU:":<20}{args.use_multi_gpu:<20}')
        print(f'  {"Multi-GPU ID:":<20}{args.devices:<20}')
    else:
        print(f'  {"Use CPU:":<20}')

    print("\033[1m" + "Save Config" + "\033[0m")
    print(f'  {"Time Stamp:":<20}{args.time_stamp:<20}')
    print(f'  {"Save Path:":<20}{args.save_path:<20}')


def log_args(args):
    logging.info(">>>>>>>Config Information<<<<<<<<<<<<<<<<<<<<<<<<<<")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
