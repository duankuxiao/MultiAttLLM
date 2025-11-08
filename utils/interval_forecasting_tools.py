import math
import numpy as np
import torch
from properscoring import crps_gaussian
from scipy.stats import norm
from torch.overrides import has_torch_function_variadic, handle_torch_function
from typing import Optional
import warnings

# NB: Keep this file in sync with enums in aten/src/ATen/core/Reduction.h
Tensor = torch.Tensor


def get_enum(reduction: str) -> int:
    if reduction == 'none':
        ret = 0
    elif reduction == 'mean':
        ret = 1
    elif reduction == 'elementwise_mean':
        warnings.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
        ret = 1
    elif reduction == 'sum':
        ret = 2
    else:
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError(f"{reduction} is not a valid value for reduction")
    return ret

# In order to support previous versions, accept boolean size_average and reduce
# and convert them into the new constants for now


# We use these functions in torch/legacy as well, in which case we'll silence the warning
def legacy_get_string(size_average: Optional[bool], reduce: Optional[bool], emit_warning: bool = True) -> str:
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True

    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        warnings.warn(warning.format(ret))
    return ret


def legacy_get_enum(size_average: Optional[bool], reduce: Optional[bool], emit_warning: bool = True) -> int:
    return get_enum(legacy_get_string(size_average, reduce, emit_warning))


class _Loss(torch.nn.Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class GaussianLikelihoodLoss(_Loss):
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float

    def __init__(self, *, full: bool = True, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super().__init__(None, None, reduction)
        self.full = full
        self.eps = eps

    '''
    Gaussian Liklihood Loss
    Args:
    target (tensor): true observations, shape (num_ts, num_periods)
    mu (tensor): mean, shape (num_ts, num_periods)
    sigma (tensor): standard deviation, shape (num_ts, num_periods)

    likelihood:
    (2 pi sigma^2)^(-1/2) exp(-(target - mu)^2 / (2 sigma^2))

    log likelihood:
    -1/2 * (log (2 pi) + 2 * log (sigma)) - 1/2 *  (target - mu)^2 / (sigma^2) + constant  
    '''
    def forward(self, input: Tensor, target: Tensor, sigma: Tensor) -> Tensor:
        return gaussian_nll_loss(input, target, sigma, full=self.full, eps=self.eps, reduction=self.reduction)


def gaussian_nll_loss(
    input: Tensor,
    target: Tensor,
    sigma: Tensor,
    full: bool = True,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> Tensor:
    r"""Gaussian negative log likelihood loss.

    See :class:`~torch.nn.GaussianNLLLoss` for details.

    Args:
        input: expectation of the Gaussian distribution.
        target: sample from the Gaussian distribution.
        sigma: tensor of positive standerd deviation(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
        full (bool, optional): include the constant term in the loss calculation. Default: ``False``.
        eps (float, optional): value added to var, for stability. Default: 1e-6.
        reduction (str, optional): specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
    if has_torch_function_variadic(input, target, sigma):
        return handle_torch_function(
            gaussian_nll_loss,
            (input, target, sigma),
            input,
            target,
            sigma,
            full=full,
            eps=eps,
            reduction=reduction,
        )

    if sigma.size() != input.size():
        if input.size()[:-1] == sigma.size():
            sigma = torch.unsqueeze(sigma, -1)
        elif input.size()[:-1] == sigma.size()[:-1] and sigma.size(-1) == 1:  # Heteroscedastic case
            pass
        else:
            raise ValueError("var is of incorrect size")

    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")
    if torch.any(sigma < 0):
        raise ValueError("var has negative entry/entries")

    sigma = sigma.clone()
    with torch.no_grad():
        sigma.clamp_(min=eps)

    # Calculate the loss
    loss = torch.log(sigma) + 0.5 * (input - target) ** 2 / sigma ** 2  # g2 gpt normal GaussianNLLLoss

    if full:
        loss += 0.5 * math.log(2 * math.pi)

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


class NegativeBinomialLoss(_Loss):
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float

    def __init__(self, *, full: bool = True, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super().__init__(None, None, reduction)
        self.full = full
        self.eps = eps

    '''
    Gaussian Liklihood Loss
    Args:
    target (tensor): true observations, shape (num_ts, num_periods)
    mu (tensor): mean, shape (num_ts, num_periods)
    sigma (tensor): standard deviation, shape (num_ts, num_periods)

    likelihood:
    (2 pi sigma^2)^(-1/2) exp(-(target - mu)^2 / (2 sigma^2))

    log likelihood:
    -1/2 * (log (2 pi) + 2 * log (sigma)) - 1/2 *  (target - mu)^2 / (sigma^2) + constant  # constant对结果没有影响
    '''
    def forward(self, input: Tensor, target: Tensor, alpha: Tensor) -> Tensor:
        return negative_binomial_loss(input, target, alpha, reduction=self.reduction)


def negative_binomial_loss(mu, ytrue, alpha,reduction='mean'):
    '''
    Negative Binomial Sample
    Args:
    ytrue (array like)
    mu (array like)
    alpha (array like)

    maximuze log l_{nb} = log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
                - 1 / alpha * log (1 + alpha * mu) + z * log (alpha * mu / (1 + alpha * mu))

    minimize loss = - log l_{nb}

    Note: torch.lgamma: log Gamma function
    '''
    batch_size, seq_len,_ = ytrue.size()
    loss = torch.lgamma(ytrue + 1. / alpha) - torch.lgamma(ytrue + 1) - torch.lgamma(1. / alpha) \
        - 1. / alpha * torch.log(1 + alpha * mu) \
        + ytrue * torch.log(alpha * mu / (1 + alpha * mu))
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def MAPE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel() + 1e-4
    ypred = np.array(ypred).ravel()
    return np.mean(np.abs((ytrue - ypred) / ytrue))


def gaussian_nll(y_true, mu, sigma):
    return np.mean(0.5 * np.log(2 * np.pi * sigma**2) + ((y_true - mu)**2) / (2 * sigma**2))


def crps_score(y_true, mu, sigma):
    return np.mean(crps_gaussian(y_true, mu, sigma))


def picp(y_true, mu, sigma, alpha=0.9):
    z = norm.ppf(1 - (1 - alpha) / 2)
    lower = mu - z * sigma
    upper = mu + z * sigma
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    return coverage


def piw(mu, sigma, alpha=0.9):
    z = norm.ppf(1 - (1 - alpha) / 2)
    width = 2 * z * sigma
    return np.mean(width)


def pinaw(y_true,mu, sigma, alpha=0.9):
    """
    mu: predicted mean, shape [N]
    sigma: predicted std, shape [N]
    y_true: true values, shape [N]
    alpha: confidence level (e.g., 0.9 for 90% interval)
    """
    # Step 1: z-value for given confidence level
    z = norm.ppf(1 - (1 - alpha) / 2)  # e.g., 1.645 for 90%

    # Step 2: construct prediction interval
    lower = mu - z * sigma
    upper = mu + z * sigma
    width = upper - lower  # shape [N]

    # Step 3: normalize by range of y_true
    y_range = np.max(y_true) - np.min(y_true)
    pinaw_value = np.mean(width) / y_range

    return pinaw_value



def gaussian_sample(mu, sigma):
    '''
    Gaussian Sample
    Args:
    ytrue (array like)
    mu (array like)
    sigma (array like): standard deviation

    gaussian maximum likelihood using log
        l_{G} (z|mu, sigma) = (2 * pi * sigma^2)^(-0.5) * exp(- (z - mu)^2 / (2 * sigma^2))
    '''
    # likelihood = (2 * np.pi * sigma ** 2) ** (-0.5) * \
    #         torch.exp((- (ytrue - mu) ** 2) / (2 * sigma ** 2))
    # return likelihood
    gaussian = torch.distributions.normal.Normal(mu, sigma)
    ypred = gaussian.rsample()
    return ypred


def negative_binomial_sample(mu, alpha):
    '''
    Negative Binomial Sample
    Args:
    ytrue (array like)
    mu (array like)
    alpha (array like)

    maximuze log l_{nb} = log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
                - 1 / alpha * log (1 + alpha * mu) + z * log (alpha * mu / (1 + alpha * mu))

    minimize loss = - log l_{nb}

    Note: torch.lgamma: log Gamma function
    '''
    var = mu + mu * mu * alpha
    ypred = mu + torch.randn(mu.size()).to(mu.device) * torch.sqrt(var)
    return ypred