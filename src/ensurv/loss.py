"""Empirical pooled-completion energy score for EnSurv."""

import torch
from .km import km_survival_at_vec, compute_psi_hat_vec


def energy_term(logT_hat):
    r"""Estimate the mean absolute distance between independent generated draws.

    :param logT_hat: Raw generated log-times, with at least two independent draws per
        observation.
    :type logT_hat: torch.Tensor of shape (n, m)

    :returns: Scalar ordered-distinct-pair average, differentiable with respect to the draws.
    :rtype: torch.Tensor

    :raises ValueError: Fewer than two draws per observation are supplied.

    .. math::

       \widehat E=\frac{1}{n m(m-1)}
       \sum_{i=1}^n\sum_{j\ne k}|U_{ij}-U_{ik}|.
    """
    n, m = logT_hat.shape
    if m < 2:
        raise ValueError('The energy score requires m>=2 independent draws.')
    distance = (logT_hat[:, :, None] - logT_hat[:, None, :]).abs()
    off_diagonal = ~torch.eye(m, dtype=torch.bool, device=logT_hat.device)
    return distance[:, off_diagonal].mean()


def survival_engression_loss(logT_hat, Y_t, D_t, km_bootstrap, lambda_energy=0.5,
                              return_components=False):
    r"""Score raw generated log-times against restricted observed and completed outcomes.

    :param logT_hat: Raw, uncapped generated log-times in the fitted model time units; m must be
        at least two.
    :type logT_hat: torch.Tensor of shape (n, m)
    :param Y_t: Positive horizon-restricted follow-up times in model time units, not their
        logarithms.
    :type Y_t: torch.Tensor of shape (n,)
    :param D_t: Restricted observation indicators. A value of 1 includes actual observed
        failures and follow-up reaching the horizon.
    :type D_t: torch.Tensor of shape (n,)
    :param km_bootstrap: Pooled bootstrap completion curves using the same time units. At least
        one is required when D_t contains zero.
    :type km_bootstrap: list[dict]
    :param lambda_energy: Coefficient subtracted times the pairwise energy term; default 0.5 is
        the energy-score coefficient in the paper. Other values change the objective.
    :type lambda_energy: float, optional
    :param return_components: If True, also return detached event_loss, censor_loss, and
        energy_term tensors for diagnostics.
    :type return_components: bool, optional

    :returns: Scalar differentiable loss, optionally paired with its unweighted component
        values.
    :rtype: torch.Tensor or tuple[torch.Tensor, dict[str, torch.Tensor]]

    .. math::

       \widehat L=\frac{1}{nm}\sum_{i,j}
       \left[D_i|\log Y_{\tau,i}-U_{ij}|+
       (1-D_i)\frac{1}{B}\sum_{b=1}^B\widehat\Psi_b(Y_{\tau,i},U_{ij})\right]
       -\lambda_{\mathrm{energy}}\widehat E.

    The historical ``event_loss`` key denotes all observed restricted outcomes,
    rather than only actual graft failures. Early-censored outcomes are completed
    using pooled KM tails. This targets the pseudo-complete working law, which need 
    not equal the true conditional failure law.
    """
    n, m = logT_hat.shape
    if n < 1 or m < 2:
        raise ValueError('A nonempty batch and m>=2 are required.')
    logY = Y_t.log()
    complete = D_t == 1
    event_loss = ((logT_hat - logY[:, None]).abs() * complete[:, None]).sum() / (n * m)
    censor_loss = logT_hat.new_zeros(())
    censored = ~complete
    if censored.any():
        if not km_bootstrap:
            raise ValueError('Early-censored observations require at least one completion curve.')
        predictions, c = logT_hat[censored], Y_t[censored]
        psi = torch.zeros_like(predictions, dtype=torch.float64)
        for km in km_bootstrap:
            denoms = km_survival_at_vec(km, c)
            psi = psi + compute_psi_hat_vec(predictions, km['time'], km['weight'], c,
                                             denoms, tau=km['tau'])
        censor_loss = (psi / len(km_bootstrap)).sum() / (n * m)
    energy = energy_term(logT_hat)
    total = event_loss + censor_loss - lambda_energy * energy
    if return_components:
        return total, {'event_loss': event_loss.detach(), 'censor_loss': censor_loss.detach(),
                       'energy_term': energy.detach()}
    return total

