import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def R2(pred, truth):
    tss = np.sum((truth - np.mean(truth)) ** 2)
    rss = np.sum((truth - pred) ** 2)
    r2 = 1 - rss / tss
    return r2


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    r2 = R2(pred, true)

    return mse, rmse, mae, mape, r2