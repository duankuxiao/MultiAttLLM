import torch
from typing import Optional, Union
import numpy as np
from scipy import optimize
import math

from utils.mask_method import generate_mcar_mask, generate_mar_mask, generate_rdo_mask


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask



def _mcar_numpy(
    X: np.ndarray,
    p: float,
) -> np.ndarray:
    assert 0 < p < 1, f"p must be in range (0, 1), but got {p}"

    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)
    mcar_missing_mask = np.asarray(np.random.rand(np.prod(X.shape)) < p)
    mcar_missing_mask = mcar_missing_mask.reshape(X.shape)
    X[mcar_missing_mask] = np.nan  # mask values selected by mcar_missing_mask
    return X


def _mcar_torch(
    X: torch.Tensor,
    p: float,
) -> torch.Tensor:
    assert 0 < p < 1, f"p must be in range (0, 1), but got {p}"

    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)
    mcar_missing_mask = torch.rand(X.shape) < p
    X[mcar_missing_mask] = torch.nan  # mask values selected by mcar_missing_mask
    return X


def mcar(
    X: Union[np.ndarray, torch.Tensor],
    p: float,
) -> Union[np.ndarray, torch.Tensor]:
    assert 0 < p < 1, f"p must be in range (0, 1), but got {p}"

    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray):
        corrupted_X = _mcar_numpy(X, p)
    elif isinstance(X, torch.Tensor):
        corrupted_X = _mcar_torch(X, p)
    else:
        raise TypeError(
            f"X must be type of list/numpy.ndarray/torch.Tensor, but got {type(X)}"
        )

    return corrupted_X


def _rdo_numpy(
    X: np.ndarray,
    p: float,
) -> np.ndarray:
    assert 0 < p < 1, f"p must be in range (0, 1), but got {p}"

    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)
    ori_shape = X.shape
    X = X.reshape(-1)
    indices = np.where(~np.isnan(X))[0].tolist()
    indices = np.random.choice(
        indices,
        round(len(indices) * p),
        replace=False,
    )
    X[indices] = np.nan
    X = X.reshape(ori_shape)
    return X


def _rdo_torch(
    X: torch.Tensor,
    p: float,
) -> torch.Tensor:
    assert 0 < p < 1, f"p must be in range (0, 1), but got {p}"

    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)
    ori_shape = X.shape
    X = X.reshape(-1)
    indices = torch.where(~torch.isnan(X))[0].tolist()
    indices = np.random.choice(
        indices,
        round(len(indices) * p),
        replace=False,
    )
    X[indices] = torch.nan
    X = X.reshape(ori_shape)
    return X


def rdo(
    X: Union[np.ndarray, torch.Tensor],
    p: float,
) -> Union[np.ndarray, torch.Tensor]:
    """Create missingness in the data by randomly drop observations.

    Parameters
    ----------
    X :
        Data vector. If X has any missing values, they should be numpy.nan.

    p :
        The proportion of the observed values that will be randomly masked as missing.
        RDO (randomly drop observations) will randomly select values from the observed values to be masked as missing.
        The number of selected observations is determined by `p` and the total number of observed values in X,
        e.g. if `p`=0.1, and there are 1000 observed values in X, then 0.1*1000=100 values will be randomly selected
        to be masked as missing. If the result is not an integer, the number of selected values will be rounded to
        the nearest.

    Returns
    -------
    corrupted_X :
        Original X with artificial missing values.
        Both originally-missing and artificially-missing values are left as NaN.

    """
    assert 0 < p < 1, f"p must be in range (0, 1), but got {p}"

    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray):
        corrupted_X = _rdo_numpy(X, p)
    elif isinstance(X, torch.Tensor):
        corrupted_X = _rdo_torch(X, p)
    else:
        raise TypeError(
            f"X must be type of list/numpy.ndarray/torch.Tensor, but got {type(X)}"
        )

    return corrupted_X


def _mar_logistic_torch(
    X: Union[np.ndarray, torch.Tensor],
    rate_obs: float,
    rate_missing: float,
) -> Union[np.ndarray, torch.Tensor]:
    def pick_coefficients(X, idxs_obs=None, idxs_nas=None, self_mask=False):
        n, d = X.shape
        if self_mask:
            coeffs = torch.randn(d)
            Wx = X * coeffs
            coeffs /= torch.std(Wx, 0)
        else:
            d_obs = len(idxs_obs)
            d_na = len(idxs_nas)
            coeffs = torch.randn(d_obs, d_na)
            Wx = X[:, idxs_obs].mm(coeffs)
            coeffs /= torch.std(Wx, 0, keepdim=True)
        return coeffs

    def fit_intercepts(X, coeffs, p, self_mask=False):
        if self_mask:
            d = len(coeffs)
            intercepts = torch.zeros(d)
            for j in range(d):

                def f(x):
                    return torch.sigmoid(X * coeffs[j] + x).mean().item() - p

                intercepts[j] = optimize.bisect(f, -50, 50)
        else:
            d_obs, d_na = coeffs.shape
            intercepts = torch.zeros(d_na)
            for j in range(d_na):

                def f(x):
                    return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p

                intercepts[j] = optimize.bisect(f, -50, 50)
        return intercepts

    assert len(X.shape) == 2, "X should be 2 dimensional"
    n, d = X.shape

    ori_type_is_np = isinstance(X, np.ndarray)
    if ori_type_is_np:
        X = torch.from_numpy(X).to(torch.float32)
    else:
        X = torch.clone(X).to(torch.float32)

    assert (
        torch.isnan(X).sum() == 0
    ), "the input X of the mar_logistic() shouldn't containing originally missing data"

    mask = torch.zeros(n, d).bool()

    # number of variables that will have no missing values (at least one variable)
    d_obs = max(int(rate_obs * d), 1)
    d_na = d - d_obs  # number of variables that will have missing values

    # Sample variables will all be observed, and the left will be with missing values
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coefficients(X, idxs_obs, idxs_nas)
    # Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, rate_missing)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)
    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps  # True means missing

    X[mask] = torch.nan

    return X.numpy() if ori_type_is_np else X


def mar_logistic(
    X: Union[torch.Tensor, np.ndarray],
    obs_rate: float,
    missing_rate: float,
) -> Union[np.ndarray, torch.Tensor]:
    """Create random missing values (MAR case) with a logistic model.
    First, a subset of the variables without missing values is randomly selected.
    Missing values will be introduced into the remaining variables according to a logistic model with random weights.
    This implementation is inspired by the tutorial
    https://rmisstastic.netlify.app/how-to/python/generate_html/how%20to%20generate%20missing%20values

    Parameters
    ----------
    X :
        A time series data vector without any missing data. Shape of [n_steps, n_features].

    obs_rate :
        The proportion of variables without missing values that will be used for fitting the logistic masking model.

    missing_rate:
        The proportion of missing values to generate for variables which will have missing values.

    Returns
    -------
    corrupted_X :
        Original X with artificial missing values.
        Both originally-missing and artificially-missing values are left as NaN.

    """
    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray) or isinstance(X, torch.Tensor):
        corrupted_X = _mar_logistic_torch(X, obs_rate, missing_rate)
    else:
        raise TypeError(
            f"X must be type of list/numpy.ndarray/torch.Tensor, but got {type(X)}"
        )

    return corrupted_X

def _mnar_x_numpy(
    X: np.ndarray,
    offset: float = 0,
) -> np.ndarray:
    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)

    n_s, n_l, n_c = X.shape
    ori_mask = ~np.isnan(X)
    mask_sum = ori_mask.sum(1)
    mask_sum[mask_sum == 0] = 1
    X_mean = np.repeat(
        ((X * ori_mask).sum(1) / mask_sum).reshape(n_s, 1, n_c), n_l, axis=1
    )
    X_std = np.repeat(
        np.sqrt(np.square((X - X_mean) * ori_mask).sum(1) / mask_sum).reshape(
            n_s, 1, n_c
        ),
        n_l,
        axis=1,
    )
    mnar_missing_mask = np.zeros_like(X)
    mnar_missing_mask[X <= (X_mean + offset * X_std)] = 1
    missing_mask = ori_mask * mnar_missing_mask
    X[missing_mask == 0] = np.nan
    return X


def _mnar_x_torch(
    X: torch.Tensor,
    offset: float = 0,
) -> torch.Tensor:
    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)

    n_s, n_l, n_c = X.shape
    ori_mask = (~torch.isnan(X)).type(torch.float32)
    mask_sum = ori_mask.sum(1)
    mask_sum[mask_sum == 0] = 1
    X_mean = ((X * ori_mask).sum(1) / mask_sum).reshape(n_s, 1, n_c).repeat(1, n_l, 1)
    X_std = (
        (((X - X_mean) * ori_mask).pow(2).sum(1) / mask_sum)
        .sqrt()
        .reshape(n_s, 1, n_c)
        .repeat(1, n_l, 1)
    )
    mnar_missing_mask = torch.zeros_like(X)
    mnar_missing_mask[X <= (X_mean + offset * X_std)] = 1
    missing_mask = ori_mask * mnar_missing_mask
    X[missing_mask == 0] = torch.nan
    return X


def mnar_x(
    X: Optional[Union[np.ndarray, torch.Tensor]],
    offset: float = 0,
) -> Union[np.ndarray, torch.Tensor]:
    """Create not-random missing values related to values themselves (MNAR-x case ot self-masking MNAR case).
    This case follows the setting in Ipsen et al. "not-MIWAE: Deep Generative Modelling with Missing Not at Random Data"
    :cite:`ipsen2021notmiwae`.

    Parameters
    ----------
    X :
        Data vector. If X has any missing values, they should be numpy.nan.

    offset :
        the weight of standard deviation. In MNAR-x case, for each time series,
        the values larger than the mean of each time series plus offset*standard deviation will be missing

    Returns
    -------
    corrupted_X :
        Original X with artificial missing values.
        Both originally-missing and artificially-missing values are left as NaN.
    """
    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray):
        corrupted_X = _mnar_x_numpy(X, offset)
    elif isinstance(X, torch.Tensor):
        corrupted_X = _mnar_x_torch(X, offset)
    else:
        raise TypeError(
            f"X must be type of list/numpy.ndarray/torch.Tensor, but got {type(X)}"
        )

    return corrupted_X

def _mnar_t_numpy(
    X: np.ndarray,
    cycle: float = 20,
    pos: float = 10,
    scale: float = 3,
) -> np.ndarray:
    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)

    n_s, n_l, n_c = X.shape
    ori_mask = (~np.isnan(X)).astype(np.float32)
    ts = np.linspace(0, 1, n_l).reshape(1, n_l, 1)
    ts = np.repeat(ts, n_s, axis=0)
    ts = np.repeat(ts, n_c, axis=2)
    intensity = np.exp(3 * np.sin(cycle * ts + pos))
    mnar_missing_mask = (np.random.rand(n_s, n_l, n_c) * scale) < intensity
    missing_mask = ori_mask * mnar_missing_mask
    X[missing_mask == 0] = np.nan
    return X


def _mnar_t_torch(
    X: torch.Tensor,
    cycle: float = 20,
    pos: float = 10,
    scale: float = 3,
) -> torch.Tensor:
    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)

    n_s, n_l, n_c = X.shape
    ori_mask = (~torch.isnan(X)).type(torch.float32)
    ts = torch.linspace(0, 1, n_l).reshape(1, n_l, 1).repeat(n_s, 1, n_c)
    intensity = torch.exp(3 * torch.sin(cycle * ts + pos))
    mnar_missing_mask = (torch.randn(X.size()).uniform_(0, 1) * scale) < intensity
    missing_mask = ori_mask * mnar_missing_mask
    X[missing_mask == 0] = torch.nan
    return X


def mnar_t(
    X: Union[np.ndarray, torch.Tensor],
    cycle: float = 20,
    pos: float = 10,
    scale: float = 3,
) -> Union[np.ndarray, torch.Tensor]:
    """Create not-random missing values related to temporal dynamics (MNAR-t case).
    In particular, the missingness is generated by an intensity function f(t) = exp(3*torch.sin(cycle*t + pos)).
    This case mainly follows the setting in https://hawkeslib.readthedocs.io/en/latest/tutorial.html.

    Parameters
    ----------
    X :
        Data vector. If X has any missing values, they should be numpy.nan.

    cycle :
        The cycle of the used intensity function

    pos :
        The displacement of the used intensity function

    scale :
        The scale number to control the missing rate


    Returns
    -------
    corrupted_X :
        Original X with artificial missing values.
        Both originally-missing and artificially-missing values are left as NaN.

    """

    if isinstance(X, list):
        X = np.asarray(X)

    if isinstance(X, np.ndarray):
        corrupted_X = _mnar_t_numpy(X, cycle, pos, scale)
    elif isinstance(X, torch.Tensor):
        corrupted_X = _mnar_t_torch(X, cycle, pos, scale)
    else:
        raise TypeError(
            f"X must be type of list/numpy.ndarray/torch.Tensor, but got {type(X)}"
        )
    return corrupted_X

def random_select_start_indices(
    feature_idx,
    step_idx,
    hit_rate,
    n_samples,
    n_steps,
    n_features,
) -> np.ndarray:
    if feature_idx is None:
        all_feature_indices = list(range(n_samples * n_features))
        all_feature_start_indices = [i * n_steps for i in all_feature_indices]
    else:
        all_feature_indices = [
            i * n_features + j for i in range(n_samples) for j in feature_idx
        ]
        all_feature_start_indices = [i * n_steps for i in all_feature_indices]


    selected_feature_start_indices = np.random.choice(
        all_feature_start_indices,
        math.ceil(len(all_feature_start_indices) * hit_rate),
        replace=hit_rate > 1,
    )
    selected_feature_start_indices = np.asarray(selected_feature_start_indices)

    step_shift = np.random.choice(
        step_idx,
        len(selected_feature_start_indices),
    )
    step_shift = np.asarray(step_shift)

    selected_start_indices = selected_feature_start_indices + step_shift
    return selected_start_indices


def _seq_missing_numpy(
    X: np.ndarray,
    p: float,
    seq_len: int,
    feature_idx: list = None,
    step_idx: list = None,
) -> np.ndarray:
    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)

    n_samples, n_steps, n_features = X.shape
    hit_rate = p * n_steps / seq_len
    start_indices = random_select_start_indices(
        feature_idx, step_idx, hit_rate, n_samples, n_steps, n_features
    )

    X = X.transpose(0, 2, 1)
    X = X.reshape(-1)
    for idx in start_indices:
        X[idx : idx + seq_len] = np.nan

    X = X.reshape(n_samples, n_features, n_steps)
    X = X.transpose(0, 2, 1)
    return X


def _seq_missing_torch(
    X: torch.Tensor,
    p: float,
    seq_len: int,
    feature_idx: list = None,
    step_idx: list = None,
) -> torch.Tensor:
    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)

    n_samples, n_steps, n_features = X.shape
    hit_rate = p * n_steps / seq_len
    start_indices = random_select_start_indices(
        feature_idx, step_idx, hit_rate, n_samples, n_steps, n_features
    )

    X = X.transpose(1, 2)
    X = X.flatten()
    for idx in start_indices:
        X[idx : idx + seq_len] = np.nan

    X = X.reshape(n_samples, n_features, n_steps)
    X = X.transpose(1, 2)
    return X


def seq_missing(
    X: Union[np.ndarray, torch.Tensor],
    p: float,
    seq_len: int,
    feature_idx: list = None,
    step_idx: list = None,
) -> Union[np.ndarray, torch.Tensor]:
    """Create subsequence missing data.

    Parameters
    ----------
    X :
        Data vector. If X has any missing values, they should be numpy.nan.

    p :
        The probability that values may be masked as missing completely at random.

    seq_len :
        The length of missing sequence.

    feature_idx :
        The indices of features for missing sequences to be corrupted.

    step_idx :
        The indices of steps for a missing sequence to start with.

    Returns
    -------
    corrupted_X :
        Original X with artificial missing values.
        Both originally-missing and artificially-missing values are left as NaN.

    """
    if isinstance(X, list):
        X = np.asarray(X)
    n_samples, n_steps, n_features = X.shape

    assert 0 < p <= 1, f"p must be in range (0, 1), but got {p}"
    assert isinstance(
        seq_len, int
    ), f"`seq_len` must be type of int, but got {type(seq_len)}"
    assert seq_len <= n_steps, f"`seq_len` must be <= {n_steps}, but got {seq_len}"

    if feature_idx is not None:
        assert isinstance(
            feature_idx, list
        ), f"`feature_idx` must be type of list, but got {type(feature_idx)}"

        assert (
            max(feature_idx) <= n_features
        ), f"values in `feature_idx` must be <= {n_features}, but got {max(feature_idx)}"

    if step_idx is not None:
        assert isinstance(
            step_idx, list
        ), f"`step_idx` must be type of list, but got {type(step_idx)}"

        assert (
            max(step_idx) <= n_steps
        ), f"values in `step_idx` must be <= {n_steps}, but got {max(step_idx)}"
        assert (
            n_steps - max(step_idx) >= seq_len
        ), f"n_steps - max(step_idx) must be >= seq_len, but got {n_steps - max(step_idx)}"
    else:
        step_idx = list(range(n_steps - seq_len + 1))

    if isinstance(X, np.ndarray):
        corrupted_X = _seq_missing_numpy(
            X,
            p,
            seq_len,
            feature_idx,
            step_idx,
        )
    elif isinstance(X, torch.Tensor):
        corrupted_X = _seq_missing_torch(
            X,
            p,
            seq_len,
            feature_idx,
            step_idx,
        )
    else:
        raise TypeError(
            f"X must be type of list/numpy.ndarray/torch.Tensor, but got {type(X)}"
        )

    return corrupted_X

def random_select_start_indices(
    block_width,
    feature_idx,
    step_idx,
    hit_rate,
    n_samples,
    n_steps,
    n_features,
) -> np.ndarray:
    all_feature_indices = [
        i * n_features + j for i in range(n_samples) for j in feature_idx
    ]


    all_feature_start_indices = [i * n_steps for i in all_feature_indices]
    selected_feature_start_indices = np.random.choice(
        all_feature_start_indices,
        math.ceil(len(all_feature_start_indices) * hit_rate),
        replace=hit_rate > 1,
    )
    selected_feature_start_indices = np.asarray(selected_feature_start_indices)

    step_shift = np.random.choice(
        step_idx,
        len(selected_feature_start_indices),
    )
    step_shift = np.asarray(step_shift)

    selected_start_indices = selected_feature_start_indices + step_shift
    selected_start_indices = [
        i + j * n_steps for i in selected_start_indices for j in range(block_width)
    ]
    return np.asarray(selected_start_indices)


def _block_missing_numpy(
    X: np.ndarray,
    factor: float,
    block_len: int,
    block_width: int,
    feature_idx: list = None,
    step_idx: list = None,
) -> np.ndarray:
    # clone X to ensure values of X out of this function not being affected
    X = np.copy(X)

    n_samples, n_steps, n_features = X.shape
    hit_rate = factor * n_steps * n_features / (block_len * block_width)
    start_indices = random_select_start_indices(
        block_width, feature_idx, step_idx, hit_rate, n_samples, n_steps, n_features
    )

    X = X.transpose(0, 2, 1)
    X = X.reshape(-1)
    for idx in start_indices:
        X[idx : idx + block_len] = np.nan

    X = X.reshape(n_samples, n_features, n_steps)
    X = X.transpose(0, 2, 1)
    return X


def _block_missing_torch(
    X: torch.Tensor,
    factor: float,
    block_len: int,
    block_width: int,
    feature_idx: list = None,
    step_idx: list = None,
) -> torch.Tensor:
    # clone X to ensure values of X out of this function not being affected
    X = torch.clone(X)

    n_samples, n_steps, n_features = X.shape
    hit_rate = factor * n_steps * n_features / (block_len * block_width)
    start_indices = random_select_start_indices(
        block_width, feature_idx, step_idx, hit_rate, n_samples, n_steps, n_features
    )

    X = X.transpose(1, 2)
    X = X.flatten()
    for idx in start_indices:
        X[idx : idx + block_len] = np.nan

    X = X.reshape(n_samples, n_features, n_steps)
    X = X.transpose(1, 2)
    return X


def block_missing(
    X: Union[np.ndarray, torch.Tensor],
    factor: float,
    block_len: int,
    block_width: int,
    feature_idx: list = None,
    step_idx: list = None,
) -> Union[np.ndarray, torch.Tensor]:
    """Create block missing data.

    Parameters
    ----------
    X :
        Data vector. If X has any missing values, they should be numpy.nan.

    factor :
        The actual missing rate of block_missing is hard to be strictly controlled.
        Hence, we use ``factor`` to help adjust the final missing rate.

    block_len :
        The length of the mask block.

    block_width :
        The width of the mask block.

    feature_idx :
        The indices of features for missing block to star with.

    step_idx :
        The indices of steps for a missing block to start with.

    Returns
    -------
    corrupted_X :
        Original X with artificial missing values.
        Both originally-missing and artificially-missing values are left as NaN.

    """
    if isinstance(X, list):
        X = np.asarray(X)
    n_samples, n_steps, n_features = X.shape

    assert isinstance(
        block_len, int
    ), f"`block_len` must be type of int, but got {type(block_len)}"
    assert block_len <= n_steps, f"`seq_len` must be <= {n_steps}, but got {block_len}"

    assert isinstance(
        block_width, int
    ), f"`block_width` must be type of int, but got {type(block_width)}"
    assert (
        block_width <= n_features
    ), f"`block_width` must be <= {n_features}, but got {block_width}"

    if feature_idx is not None:
        assert isinstance(
            feature_idx, list
        ), f"`feature_idx` must be type of list, but got {type(feature_idx)}"

        assert (
            max(feature_idx) <= n_features
        ), f"values in `feature_idx` must be <= {n_features}, but got {max(feature_idx)}"
    else:
        feature_idx = list(range(n_features - block_width + 1))

    if step_idx is not None:
        assert isinstance(
            step_idx, list
        ), f"`step_idx` must be type of list, but got {type(step_idx)}"

        assert (
            max(step_idx) <= n_steps
        ), f"values in `step_idx` must be <= {n_steps}, but got {max(step_idx)}"
        assert (
            n_steps - max(step_idx) >= block_len
        ), f"n_steps - max(step_idx) must be >= block_len, but got {n_steps - max(step_idx)}"
    else:
        step_idx = list(range(n_steps - block_len + 1))

    if isinstance(X, np.ndarray):
        corrupted_X = _block_missing_numpy(
            X,
            factor,
            block_len,
            block_width,
            feature_idx,
            step_idx,
        )
    elif isinstance(X, torch.Tensor):
        corrupted_X = _block_missing_torch(
            X,
            factor,
            block_len,
            block_width,
            feature_idx,
            step_idx,
        )
    else:
        raise TypeError(
            f"X must be type of list/numpy.ndarray/torch.Tensor, but got {type(X)}"
        )

    return corrupted_X


def mask_custom(X_ori, mask_rate=0.1,method='mcar',f_dim=4,seed=4213,always_obs=4,targets_only=False):
    # grind the dataset with MCAR pattern, 10% missing probability, and using 0 to fill missing values
    if method == 'mcar':
        # X_with_mask_data = mcar(X_ori, p=mask_rate)  每个特征量随机在不同的时间缺失
        X_with_mask_data = generate_mcar_mask(X_ori, missing_rate=mask_rate,f_dim=f_dim,seed=seed,tail_targets_only=targets_only)
    elif method == 'mar':
        # grind the dataset with MAR pattern  部分特征量随机在不同的时间缺失
        # X_with_mask_data = mar_logistic(X_ori[:, 0, :], obs_rate=mask_rate, missing_rate=mask_rate)
        X_with_mask_data = generate_mar_mask(X_ori,obs_rate=1.0,missing_rate=mask_rate,f_dim=f_dim,seed=seed,always_obs=always_obs)
    elif method == 'rdo':
        # grind the dataset with randomly drop observations pattern   每个特征量随机在同一时间缺失
        # X_with_mask_data = rdo(X_ori, p=mask_rate)
        X_with_mask_data = generate_rdo_mask(X_ori, row_drop_rate=mask_rate, f_dim=f_dim, seed=seed, tail_targets_only=targets_only)
    else:
        raise ValueError('method must be mcar or mar or rdo or seq or block_missing')
    mask = (np.isnan(X_with_mask_data) ^ np.isnan(X_ori)) ^ 1
    inp = X_ori.masked_fill(mask == 0, 0)
    return X_with_mask_data, mask, inp

@torch.no_grad()
def masked_standardize_3d(
    x_enc: torch.Tensor,          # [B,T,C]
    mask: torch.Tensor,           # [B,T,1] 或 [B,T,C]，1=观测，0=缺失
    eps: float = 1e-5,            # 数值稳定项
    fill_missing_zero: bool = True,  # 归一化后是否把缺失位置置 0
):
    """
    返回:
      x_norm: [B,T,C]  归一化后的输入
      means : [B,1,C]  每样本每特征的时间均值
      stdev : [B,1,C]  每样本每特征的时间标准差
    """
    assert x_enc.ndim == 3 and mask.ndim == 3, "x_enc, mask 必须是 [B,T,C] 与 [B,T,1/ C]"
    B, T, C = x_enc.shape

    # 将 mask 扩展到与 x_enc 同维度
    m = mask
    if m.shape[-1] == 1 and C != 1:
        m = m.expand(-1, -1, C)
    elif m.shape[-1] != C:
        raise ValueError(f"mask 最后一维需为 1 或 {C}，当前 {m.shape[-1]}")

    x = x_enc.float()
    m = m.to(dtype=x.dtype)

    # --- 统计（时间维度）---
    num   = (x * m).sum(dim=1)          # [B,C]
    denom = m.sum(dim=1)                # [B,C]
    valid = denom > 0                   # 该样本-特征是否有观测

    # 均值：对 valid 做除法，对无观测回退到全局均值
    means_bt = torch.zeros_like(num)
    means_bt[valid] = num[valid] / denom[valid].clamp_min(1.0)

    if not valid.all():
        # 全局（跨 batch/时间）的掩码均值作为回退
        g_num   = num.sum(dim=0)                        # [C]
        g_denom = denom.sum(dim=0).clamp_min(1.0)       # [C]
        g_mean  = g_num / g_denom                       # [C]
        means_bt[~valid] = g_mean.unsqueeze(0).expand_as(means_bt)[~valid]

    means = means_bt.unsqueeze(1).detach()  # [B,1,C]

    # 去中心化
    x_center = x - means

    # 方差（用掩码）：E[(x-μ)^2]，对无观测回退到全局方差
    sq_num = ((x_center * x_center) * m).sum(dim=1)     # [B,C]
    var_bt = torch.zeros_like(sq_num)
    var_bt[valid] = sq_num[valid] / denom[valid].clamp_min(1.0)

    if not valid.all():
        g_sq_num = sq_num.sum(dim=0)                    # [C]
        g_var    = g_sq_num / g_denom                   # [C]（用上面 g_denom）
        var_bt[~valid] = g_var.unsqueeze(0).expand_as(var_bt)[~valid]

    stdev = torch.sqrt(var_bt + eps).unsqueeze(1).detach()  # [B,1,C]

    # 归一化
    x_norm = x_center / stdev

    # 可选：把缺失位置置 0（常见做法，避免模型“看到”无意义值）
    if fill_missing_zero:
        x_norm = x_norm.masked_fill(m == 0, 0.0)

    # 恢复到原 dtype
    x_norm = x_norm.to(dtype=x_enc.dtype)

    return x_norm, means, stdev