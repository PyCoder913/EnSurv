"""Validation and restriction of observed survival outcomes."""

import numpy as np
import torch


def _validate_outcomes(Y, Delta, device=None):
    r"""Validate original follow-up times and binary failure indicators.

    :param Y: Observed times, all finite and strictly positive.
    :type Y: array-like of shape (n,)
    :param Delta: Original indicators: 1 for observed failure and 0 for right censoring.
    :type Delta: array-like of shape (n,)
    :param device: Device on which float64 outcome tensors are constructed. None uses PyTorch
        device inference/defaults.
    :type device: torch.device, str or None

    :returns: Validated float64 time and indicator tensors. Failure indicators retain their
        original meaning.
    :rtype: tuple[torch.Tensor, torch.Tensor]

    :raises ValueError: The arrays are empty, misaligned, nonfinite, nonpositive in time, or nonbinary in event status.
    """
    y = torch.as_tensor(Y, dtype=torch.float64, device=device)
    delta = torch.as_tensor(Delta, dtype=torch.float64, device=device)
    if y.ndim != 1 or delta.shape != y.shape or y.numel() == 0:
        raise ValueError('Y and Delta must be nonempty one-dimensional arrays of equal length.')
    if not torch.isfinite(y).all() or (y <= 0).any():
        raise ValueError('Observed times must be finite and strictly positive for log-time training.')
    if not torch.isfinite(delta).all() or not ((delta == 0) | (delta == 1)).all():
        raise ValueError('Delta must contain only 0 (censored) and 1 (observed failure).')
    return y, delta


def resolve_tau(Y, Delta, tau):
    r"""Resolve the restriction horizon from a numeric value or training failures.

    :param Y: Original training follow-up times in the desired horizon units.
    :type Y: array-like of shape (n,)
    :param Delta: Original training failure indicators, with 1 denoting an observed failure.
    :type Delta: array-like of shape (n,)
    :param tau: Positive fixed horizon, or 'max_event' to use the largest observed training
        failure time.
    :type tau: float or {'max_event'}

    :returns: Resolved horizon and its source label: 'fixed' or 'max_event_training_sample'.
    :rtype: tuple[float, str]

    :raises ValueError: The horizon is invalid, no failure exists for the maximum-event option, or no training observation reaches the horizon.

    The caller must supply training data only. A sample-dependent maximum failure time is an
    experimental horizon choice.
    """
    y, delta = _validate_outcomes(Y, Delta)
    if isinstance(tau, str):
        if tau != 'max_event':
            raise ValueError("tau must be a positive number or 'max_event'.")
        failures = y[delta == 1]
        if failures.numel() == 0:
            raise ValueError("tau='max_event' needs an observed failure; supply a fixed horizon instead.")
        value = float(failures.max().item())
        source = 'max_event_training_sample'
    else:
        if tau is None or isinstance(tau, (bool, np.bool_)):
            raise ValueError("Supply a positive horizon or tau='max_event'.")
        value = float(tau)
        source = 'fixed'
    if not np.isfinite(value) or value <= 0:
        raise ValueError('tau must be finite and strictly positive.')
    if not (y >= value).any():
        raise ValueError('No training observation reaches tau; choose a horizon supported by training follow-up.')
    return value, source


def apply_tau_restriction(Y, Delta, tau):
    r"""Construct horizon-restricted outcomes without relabeling original failures.

    :param Y: Original positive follow-up times.
    :type Y: array-like of shape (n,)
    :param Delta: Original binary failure indicators.
    :type Delta: array-like of shape (n,)
    :param tau: Resolved positive horizon in the same units as Y; it is not estimated here.
    :type tau: float

    :returns: Float64 restricted times and indicators of whether the restricted outcome is
        observed.
    :rtype: tuple[torch.Tensor, torch.Tensor]

    .. math::

       Y_\tau = \min(Y,\tau), \qquad
       D = 1-(1-\Delta)\mathbf{1}\{Y<\tau\}.

    Follow-up reaching the horizon gives :math:`D=1` even without an observed
    graft failure. Only censoring strictly before the horizon gives :math:`D=0`.
    """
    y, delta = _validate_outcomes(Y, Delta, device=getattr(Y, 'device', None))
    value = float(tau)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('tau must be finite and strictly positive.')
    y_tau = torch.minimum(y, y.new_tensor(value))
    D = torch.where(y >= value, torch.ones_like(delta), delta)
    return y_tau, D

