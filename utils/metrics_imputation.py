"""
Evaluation metrics related to error calculation (like in tasks regression, imputation etc).
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from typing import Union, Optional

import numpy as np
import pandas as pd
import torch


def _check_inputs(
    predictions: Union[np.ndarray, torch.Tensor, list],
    targets: Union[np.ndarray, torch.Tensor, list],
    masks: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
    check_shape: bool = True,
):
    # check type
    assert isinstance(predictions, type(targets)), (
        f"types of `predictions` and `targets` must match, but got"
        f"`predictions`: {type(predictions)}, `target`: {type(targets)}"
    )
    lib = np if isinstance(predictions, np.ndarray) else torch
    # check shape
    prediction_shape = predictions.shape
    target_shape = targets.shape
    if check_shape:
        assert (
            prediction_shape == target_shape
        ), f"shape of `predictions` and `targets` must match, but got {prediction_shape} and {target_shape}"
    # check NaN
    assert not lib.isnan(predictions).any(), "`predictions` mustn't contain NaN values, but detected NaN in it"
    assert not lib.isnan(targets).any(), "`targets` mustn't contain NaN values, but detected NaN in it"

    if masks is not None:
        # check type
        assert isinstance(masks, type(targets)), (
            f"types of `masks`, `predictions`, and `targets` must match, but got"
            f"`masks`: {type(masks)}, `targets`: {type(targets)}"
        )
        # check shape, masks shape must match targets
        mask_shape = masks.shape
        assert mask_shape == target_shape, (
            f"shape of `masks` must match `targets` shape, "
            f"but got `mask`: {mask_shape} that is different from `targets`: {target_shape}"
        )
        # check NaN
        assert not lib.isnan(masks).any(), "`masks` mustn't contain NaN values, but detected NaN in it"

    return lib


def calc_mae(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate the Mean Absolute Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import calc_mae
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> mae = calc_mae(predictions, targets)

    mae = 0.6 here, the error is from the 3rd and 5th elements and is :math:`|3-1|+|5-6|=3`, so the result is 3/5=0.6.

    If we want to prevent some values from MAE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> mae = calc_mae(predictions, targets, masks)

    mae = 0.5 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|=1`,
    so the result is 1/2=0.5.

    """
    # check shapes and values of inputs
    lib = _check_inputs(predictions, targets, masks)

    if masks is not None:
        return lib.sum(lib.abs(predictions - targets) * masks) / (lib.sum(masks) + 1e-12)
    else:
        return lib.mean(lib.abs(predictions - targets))

def calc_mape(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[float, torch.Tensor]:
    lib = _check_inputs(predictions, targets, masks)

    # 避免除以0（例如目标为0时）
    epsilon = np.finfo(np.float64).eps
    percentage_error = lib.abs((predictions - targets) / lib.maximum(lib.abs(targets), epsilon))

    if masks is not None:
        return (percentage_error * masks).sum() / (lib.sum(masks) + 1e-12)
    else:
        return percentage_error.mean()


def calc_mse(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate the Mean Square Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import calc_mse
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> mse = calc_mse(predictions, targets)

    mse = 1 here, the error is from the 3rd and 5th elements and is :math:`|3-1|^2+|5-6|^2=5`, so the result is 5/5=1.

    If we want to prevent some values from MSE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> mse = calc_mse(predictions, targets, masks)

    mse = 0.5 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|^2=1`,
    so the result is 1/2=0.5.

    """
    # check shapes and values of inputs
    lib = _check_inputs(predictions, targets, masks)

    if masks is not None:
        return lib.sum(lib.square(predictions - targets) * masks) / (lib.sum(masks) + 1e-12)
    else:
        return lib.mean(lib.square(predictions - targets))


def calc_rmse(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate the Root Mean Square Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import calc_rmse
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> rmse = calc_rmse(predictions, targets)

    rmse = 1 here, the error is from the 3rd and 5th elements and is :math:`|3-1|^2+|5-6|^2=5`,
    so the result is :math:`\\sqrt{5/5}=1`.

    If we want to prevent some values from RMSE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> rmse = calc_rmse(predictions, targets, masks)

    rmse = 0.707 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|^2=1`,
    so the result is :math:`\\sqrt{1/2}=0.5`.

    """
    # don't have to check types and NaN here, since calc_mse() will do it
    lib = np if isinstance(predictions, np.ndarray) else torch
    return lib.sqrt(calc_mse(predictions, targets, masks))


def calc_mre(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[float, torch.Tensor]:
    """Calculate the Mean Relative Error between ``predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    Examples
    --------

    >>> import numpy as np
    >>> from pypots.utils.metrics import calc_mre
    >>> targets = np.array([1, 2, 3, 4, 5])
    >>> predictions = np.array([1, 2, 1, 4, 6])
    >>> mre = calc_mre(predictions, targets)

    mre = 0.2 here, the error is from the 3rd and 5th elements and is :math:`|3-1|+|5-6|=3`,
    so the result is :math:`\\sqrt{3/(1+2+3+4+5)}=1`.

    If we want to prevent some values from MRE calculation, e.g. the first three elements here,
    we can use ``masks`` to filter out them:

    >>> masks = np.array([0, 0, 0, 1, 1])
    >>> mre = calc_mre(predictions, targets, masks)

    mre = 0.111 here, the first three elements are ignored, the error is from the 5th element and is :math:`|5-6|^2=1`,
    so the result is :math:`\\sqrt{1/2}=0.5`.

    """
    # check shapes and values of inputs
    lib = _check_inputs(predictions, targets, masks)

    if masks is not None:
        return lib.sum(lib.abs(predictions - targets) * masks) / (lib.sum(lib.abs(targets * masks)) + 1e-12)
    else:
        return lib.sum(lib.abs(predictions - targets)) / (lib.sum(lib.abs(targets)) + 1e-12)


def calc_quantile_loss(predictions, targets, q: float, eval_points) -> float:
    quantile_loss = 2 * torch.sum(
        torch.abs((predictions - targets) * eval_points * ((targets <= predictions) * 1.0 - q))
    )
    return quantile_loss


def calc_quantile_crps(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Union[np.ndarray, torch.Tensor],
    scaler_mean=0,
    scaler_stddev=1,
) -> float:
    """Continuous rank probability score for distributional predictions.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        Only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    scaler_mean:
        Mean value of the scaler used to scale the data.

    scaler_stddev:
        Standard deviation value of the scaler used to scale the data.

    Returns
    -------
    CRPS :
        Value of continuous rank probability score.

    """
    # check shapes and values of inputs
    _ = _check_inputs(predictions, targets, masks, check_shape=False)

    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)

    targets = targets * scaler_stddev + scaler_mean
    predictions = predictions * scaler_stddev + scaler_mean

    quantiles = np.arange(0.05, 1.0, 0.05)
    denominator = torch.sum(torch.abs(targets * masks))
    CRPS = torch.tensor(0.0)
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(predictions)):
            q_pred.append(torch.quantile(predictions[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = calc_quantile_loss(targets, q_pred, quantiles[i], masks)
        CRPS += q_loss / denominator
    return CRPS.item() / len(quantiles)


def calc_quantile_crps_sum(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    masks: Union[np.ndarray, torch.Tensor],
    scaler_mean=0,
    scaler_stddev=1,
) -> float:
    """Sum continuous rank probability score for distributional predictions.

    Parameters
    ----------
    predictions :
        The prediction data to be evaluated.

    targets :
        The target data for helping evaluate the predictions.

    masks :
        The masks for filtering the specific values in inputs and target from evaluation.
        Only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.

    scaler_mean:
        Mean value of the scaler used to scale the data.

    scaler_stddev:
        Standard deviation value of the scaler used to scale the data.

    Returns
    -------
    CRPS :
        Sum value of continuous rank probability score.

    """
    # check shapes and values of inputs
    _ = _check_inputs(predictions, targets, masks, check_shape=False)

    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)

    masks = masks.mean(-1)
    targets = targets * scaler_stddev + scaler_mean
    targets = targets.sum(-1)
    predictions = predictions * scaler_stddev + scaler_mean

    quantiles = np.arange(0.05, 1.0, 0.05)
    denominator = torch.sum(torch.abs(targets * masks))
    CRPS = torch.tensor(0.0)
    for i in range(len(quantiles)):
        q_pred = torch.quantile(predictions.sum(-1), quantiles[i], dim=1)
        q_loss = calc_quantile_loss(targets, q_pred, quantiles[i], masks)
        CRPS += q_loss / denominator
    return CRPS.item() / len(quantiles)


def results_evaluation_imputation(y_test_seq, y_pred_seq, mask):
    mae = calc_mae(targets=y_test_seq, predictions=y_pred_seq,masks=mask)
    mape = calc_mape(targets=y_test_seq, predictions=y_pred_seq,masks=mask)
    mse = calc_mse(targets=y_test_seq, predictions=y_pred_seq,masks=mask)
    rmse = calc_rmse(targets=y_test_seq, predictions=y_pred_seq,masks=mask)
    mre = calc_mre(targets=y_test_seq, predictions=y_pred_seq,masks=mask)
    return [mse, rmse, mae, mape, mre]


def interpolate_nan_matrix(matrix, method='linear', axis=0, order=None):
    """
    对带有 NaN 值的矩阵进行插值，返回插值后的完整矩阵。

    参数：
    - matrix (ndarray): 输入的带有 NaN 值的矩阵。
    - method (str): 插值方法，默认是 'linear'。
                    其他可选方法包括 'polynomial', 'spline' 等。
    - axis (int): 插值方向，0 表示按列插值，1 表示按行插值。
    - order (int): 多项式或样条插值的阶数，仅在方法为 'polynomial' 或 'spline' 时使用。

    返回：
    - interpolated_matrix (ndarray): 插值后的完整矩阵。
    """
    # 将输入矩阵转换为 DataFrame，以便使用 pandas 的插值功能
    df = pd.DataFrame(matrix)

    if method in ['polynomial', 'spline'] and order is not None:
        if method == 'spline':
            assert 1 <= order <= 5, "order should be more than 1 and less than 5"
        df = df.interpolate(method=method, axis=axis, order=order)
    else:
        df = df.interpolate(method=method)
    df_interpolated = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    # 将 DataFrame 转换回 ndarray
    interpolated_matrix = df_interpolated.to_numpy()
    return interpolated_matrix