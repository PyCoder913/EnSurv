"""Pooled bootstrap Kaplan-Meier completion and vectorized residual tails."""

import numpy as np
import torch
from .horizon import _validate_outcomes


def kaplan_meier_batched(Y, Delta, indices, close_tail=True, device=None,
                         tau=None, bootstrap_batch_size=4):
    r"""Estimate pooled bootstrap Kaplan-Meier completion laws at a shared horizon.

    :param Y: Original, unrestricted positive follow-up times. Their units must match tau.
    :type Y: array-like of shape (n,)
    :param Delta: Original failure indicators; horizon-restricted indicators must not be
        substituted.
    :type Delta: array-like of shape (n,)
    :param indices: Bootstrap row indices. Each row defines an n-observation resample; indices
        must lie between 0 and n - 1.
    :type indices: array-like of shape (B, n)
    :param close_tail: If True, append the residual pre-horizon survival mass at tau, producing
        a proper completion law. If False, return only event-jump masses.
    :type close_tail: bool, optional
    :param device: Computation device. None uses the input time tensor device when available,
        otherwise CPU.
    :type device: torch.device, str or None, optional
    :param tau: Positive shared horizon in Y units. Required for tail closure. When omitted with
        close_tail=False, no finite horizon cutoff is applied.
    :type tau: float or None, optional
    :param bootstrap_batch_size: Number of bootstrap resamples processed together to bound
        temporary memory; default is 4.
    :type bootstrap_batch_size: int, optional

    :returns: One dictionary per resample with time, weight, jump_time, surv_after, tau, and
        terminal_mass entries.
    :rtype: list[dict]

    :raises ValueError: Outcomes, index dimensions/range, batch size, or the closure horizon are invalid.

    At distinct event times :math:`t_j<\tau`, with :math:`d_j` failures and
    :math:`r_j` at risk, the survival and event masses are

    .. math::

       \widehat S(t_j)=\prod_{k\leq j}\left(1-\frac{d_k}{r_k}\right),
       \qquad w_j=\widehat S(t_j-)\frac{d_j}{r_j}.

    Ties are aggregated exactly and risk sets include observations with
    :math:`Y\geq t_j`. Events at or after the horizon do not create pre-horizon
    jumps. ``time`` and ``weight`` describe completion support and masses;
    ``jump_time`` and ``surv_after`` retain event jumps before artificial closure;
    ``terminal_mass`` is the residual survival before closure.
    """
    if device is None:
        device = getattr(Y, 'device', torch.device('cpu'))
    y, delta = _validate_outcomes(Y, Delta, device=device)
    idx = torch.as_tensor(indices, dtype=torch.long, device=device)
    if idx.ndim != 2 or idx.shape[1] != y.numel() or idx.shape[0] < 1:
        raise ValueError('indices must have shape (B,n), with B>=1 and the original n.')
    if (idx < 0).any() or (idx >= y.numel()).any():
        raise ValueError('Bootstrap indices are outside the observed sample.')
    if bootstrap_batch_size < 1:
        raise ValueError('bootstrap_batch_size must be positive.')
    if close_tail and (tau is None or not np.isfinite(float(tau)) or float(tau) <= 0):
        raise ValueError('A numeric tau is required to close KM tails at a common horizon.')
    horizon = float(tau) if tau is not None else float('inf')
    curves = []
    n = y.numel()
    for lo in range(0, idx.shape[0], bootstrap_batch_size):
        ids = idx[lo:lo + bootstrap_batch_size]
        times, order = torch.sort(y[ids], dim=1, stable=True)
        events = torch.gather(delta[ids], 1, order)
        first = torch.cat((torch.ones((len(ids), 1), dtype=torch.bool, device=device),
                           times[:, 1:] != times[:, :-1]), dim=1)
        last = torch.cat((times[:, :-1] != times[:, 1:],
                          torch.ones((len(ids), 1), dtype=torch.bool, device=device)), dim=1)
        positions = torch.arange(n, device=device).expand_as(times)
        group_start = torch.cummax(torch.where(first, positions, torch.zeros_like(positions)), dim=1).values
        cumulative_events = events.cumsum(dim=1)
        for row in range(len(ids)):
            end = last[row]
            group_times = times[row, end]
            end_events = cumulative_events[row, end]
            counts = end_events - torch.cat((end_events.new_zeros(1), end_events[:-1]))
            risk = n - group_start[row, end]
            before_tau = group_times < horizon
            group_times, counts, risk = group_times[before_tau], counts[before_tau], risk[before_tau]
            after = torch.cumprod(1.0 - counts / risk, dim=0)
            before = torch.cat((after.new_ones(1), after[:-1])) if after.numel() else after
            jumps = counts > 0
            jump_times = group_times[jumps]
            masses = (before * counts / risk)[jumps]
            jump_survival = after[jumps]
            terminal = after[-1] if after.numel() else y.new_tensor(1.)
            support, weights = jump_times, masses
            if close_tail:
                support = torch.cat((support, y.new_tensor([horizon])))
                weights = torch.cat((weights, terminal.reshape(1)))
            curves.append({'time': support, 'weight': weights,
                           'jump_time': jump_times, 'surv_after': jump_survival,
                           'tau': horizon, 'terminal_mass': terminal})
    return curves


def km_survival_at_vec(km_fit, C_array):
    r"""Evaluate pre-closure KM survival after all event jumps at each query time.

    :param km_fit: A completion-curve dictionary containing jump_time and surv_after from
        kaplan_meier_batched.
    :type km_fit: dict
    :param C_array: Censoring query times in the same units as the curve. Its device determines
        the output device.
    :type C_array: torch.Tensor of shape (n_c,)

    :returns: Float64 survival denominators; values before the first event jump are one.
    :rtype: torch.Tensor of shape (n_c,)

    A query exactly at an event time includes that event jump. The artificial terminal mass at
    the horizon is excluded from this survival lookup.
    """
    c = torch.as_tensor(C_array, dtype=torch.float64, device=C_array.device)
    times = km_fit['jump_time'].to(device=c.device, dtype=c.dtype)
    survival = km_fit['surv_after'].to(device=c.device, dtype=c.dtype)
    ids = torch.searchsorted(times, c.contiguous(), right=True)
    result = torch.ones_like(c)
    valid = ids > 0
    result[valid] = survival[ids[valid] - 1]
    return result


def compute_psi_hat_vec(predictions, km_time, km_weight, censor_times, denoms, tau=None):
    r"""Evaluate conditional pooled-tail absolute deviations using prefix sums.

    :param predictions: Raw generated log-times for the early-censored observations.
    :type predictions: torch.Tensor of shape (n_c, m)
    :param km_time: Sorted positive completion support in the same time units as censor_times,
        including the horizon atom when the tail is closed.
    :type km_time: torch.Tensor of shape (L,)
    :param km_weight: Probability masses corresponding to km_time.
    :type km_weight: torch.Tensor of shape (L,)
    :param censor_times: Early-censoring times; only support strictly greater than each time
        contributes.
    :type censor_times: torch.Tensor of shape (n_c,)
    :param denoms: Pre-closure KM survival at each censoring time, calculated after tied event
        jumps.
    :type denoms: torch.Tensor of shape (n_c,)
    :param tau: Numeric horizon in support-time units, used for the zero-denominator fallback.
        None infers the last support time; empty support requires an explicit value.
    :type tau: float or None, optional

    :returns: Float64 conditional absolute deviations on the prediction device.
    :rtype: torch.Tensor of shape (n_c, m)

    For positive denominators, the calculation is

    .. math::

       \widehat\Psi(c,u)=
       \frac{\sum_{j:t_j>c}w_j|\log t_j-u|}{\widehat S(c)}.

    Zero denominators use :math:`|\log\tau-u|`. Each bootstrap replicate is
    normalized separately before the loss averages replicates. Prefix sums avoid
    allocating a tensor with one dimension for every support point and draw.
    """
    device = predictions.device
    times = km_time.to(device=device, dtype=torch.float64)
    weights = km_weight.to(device=device, dtype=torch.float64)
    c = censor_times.to(device=device, dtype=torch.float64)
    denom = denoms.to(device=device, dtype=torch.float64)
    if tau is None:
        if times.numel() == 0:
            raise ValueError('tau is needed for an empty completion support.')
        tau = float(times[-1].item())
    a = predictions.new_tensor(np.log(float(tau)), dtype=torch.float64)
    fallback = (predictions - a).abs()
    if times.numel() == 0:
        return fallback
    z = times.log()
    W = torch.cat((weights.new_zeros(1), weights.cumsum(0)))
    ZW = torch.cat((weights.new_zeros(1), (weights * z).cumsum(0)))
    lower = torch.searchsorted(times, c.contiguous(), right=True)
    split = torch.searchsorted(z, predictions.to(torch.float64).contiguous(), right=True)
    split = torch.maximum(split, lower[:, None])
    left_w = W[split] - W[lower][:, None]
    left_z = ZW[split] - ZW[lower][:, None]
    right_w = W[-1] - W[split]
    right_z = ZW[-1] - ZW[split]
    numerator = (predictions * (left_w - right_w) - left_z + right_z).clamp_min(0.)
    valid = denom > 0
    safe = torch.where(valid, denom, torch.ones_like(denom))
    return torch.where(valid[:, None], numerator / safe[:, None], fallback)

