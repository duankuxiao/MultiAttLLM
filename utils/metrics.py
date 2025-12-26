import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from utils.interval_forecasting_tools import gaussian_nll, crps_score, picp, piw, pinaw
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, average_precision_score


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
    mape = calc_mape_without_outliers(y_true=y_test_seq, y_pred=y_pred_seq)
    rae = RAE(y_test_seq, y_pred_seq)
    r2 = r2_score(y_true=y_test_seq, y_pred=y_pred_seq,multioutput='uniform_average')  # multioutput='variance_weighted' 'uniform_average'
    corr = empirical_correlation_coefficient(y_true=y_test_seq,y_pred=y_pred_seq)
    return [mse, rmse,nrmse, mae,mape,rae, r2,corr]


def calc_mape_without_outliers(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 2.0,
    eps: float = 1e-8
) -> float:
    """
    计算去除异常值后的 MAPE

    参数：
    - predictions: 模型预测值
    - targets: 真实值
    - threshold: 最大允许的百分比误差（如 1.0 表示 100%）
    - eps: 防止除零

    返回：
    - 去除异常值后的 MAPE
    """

    # 计算每个样本的百分比误差
    percentage_error = np.abs((y_pred - y_true) / (np.abs(y_true) + eps))

    # 识别“非异常值”（误差小于设定阈值）
    mask = percentage_error < threshold

    # 打印异常点个数（可选）
    num_outliers = (~mask).sum()
    print(f"被剔除的异常样本数量：{num_outliers} / {len(y_true)}")

    # 筛选有效位置并计算 MAPE
    if mask.sum() == 0:
        return np.nan  # 全部是异常点，返回 NaN
    return np.mean(percentage_error[mask])


def results_evaluation_classification(y_test_seq, y_pred_seq):
    acc = accuracy_score(y_true=y_test_seq, y_pred=y_pred_seq)
    f1 = f1_score(y_true=y_test_seq, y_pred=y_pred_seq,average='macro')
    precision = precision_score(y_true=y_test_seq, y_pred=y_pred_seq,average='macro')
    recall = recall_score(y_true=y_test_seq, y_pred=y_pred_seq,average='macro')
    return acc, precision, f1, recall



def results_probability_forecast_evaluation(y_true, mu, sigma = None,y_lower = None, y_upper = None):
    if sigma is not None:
        nll = gaussian_nll(y_true, mu, sigma)
        crps = crps_score(y_true, mu, sigma)
        picp90 = picp(y_true, mu, sigma, alpha=0.9)
        picp80 = picp(y_true, mu, sigma, alpha=0.8)
        picp70 = picp(y_true, mu, sigma, alpha=0.7)
        piw90 = piw(mu, sigma, alpha=0.9)
        piw80 = piw(mu, sigma, alpha=0.8)
        piw70 = piw(mu, sigma, alpha=0.7)
        pinaw90 = pinaw(y_true, mu, sigma, alpha=0.9)
        pinaw80 = pinaw(y_true,mu, sigma, alpha=0.8)
        pinaw70 = pinaw(y_true,mu, sigma, alpha=0.7)
        return [nll, crps, picp90, picp80, picp70, piw90, piw80, piw70, pinaw90, pinaw80, pinaw70]
    else:
        picp90 = np.mean((y_true >= y_lower) & (y_true <= y_upper))
        piw90 = np.mean(y_upper - y_lower)
        return [picp90, piw90]



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
