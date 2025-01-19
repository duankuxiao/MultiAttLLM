import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


def empirical_correlation_coefficient(y_true, y_pred):
    """
    Calculate the Empirical Correlation Coefficient (CORR) for time series forecasting.

    :param y_true: numpy array of true values
    :param y_pred: numpy array of predicted values
    :return: CORR value
    """
    # Ensure that y_true and y_pred are numpy arrays and have the same length
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, "The true and predicted values must have the same shape."

    # Calculate the mean of the true and predicted values
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)

    # Calculate the numerator and the denominator of the CORR formula
    numerator = np.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
    denominator = np.sqrt(np.sum((y_true - y_true_mean)**2) * np.sum((y_pred - y_pred_mean)**2))

    # Calculate the CORR value
    corr = numerator / denominator

    return corr


def results_evaluation(y_test_seq, y_pred_seq):
    mse = mean_squared_error(y_true=y_test_seq, y_pred=y_pred_seq)
    rmse = np.sqrt(mse)
    nrmse,_ = NRMSE(y_test_seq, y_pred_seq,rmse)
    mae = mean_absolute_error(y_true=y_test_seq, y_pred=y_pred_seq)
    mape = mean_absolute_percentage_error(y_true=y_test_seq, y_pred=y_pred_seq)
    rae = RAE(y_test_seq, y_pred_seq)
    r2 = r2_score(y_true=y_test_seq, y_pred=y_pred_seq,multioutput='uniform_average')  # multioutput='variance_weighted' 'uniform_average'
    corr = empirical_correlation_coefficient(y_true=y_test_seq,y_pred=y_pred_seq)
    return [mse, rmse,nrmse, mae,mape,rae, r2,corr]

def NRMSE(y_true, y_pred,rmse):
    # 标准化 RMSE (可以选择用真实值的范围或平均值)
    nrmse_range = rmse / (np.max(y_true) - np.min(y_true))  # 使用范围标准化
    nrmse_mean = rmse / np.mean(y_true)  # 使用平均值标准化
    return nrmse_range, nrmse_mean

def RAE(y_true, y_pred):
    # 计算 MAE
    mae = mean_absolute_error(y_true, y_pred)

    # 计算基准模型误差，即使用均值作为预测值时的误差
    y_mean = np.mean(y_true)
    mae_baseline = np.mean(np.abs(y_true - y_mean))

    # 计算 RAE
    rae = mae / mae_baseline
    return rae


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    return mae, mse, rmse, mape, mspe
