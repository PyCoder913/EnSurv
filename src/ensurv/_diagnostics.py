"""Internal parameter-group and structural-weight diagnostics."""

import numpy as np
import torch.nn as nn


def _get_param_groups(net):
    r"""Collect parameter names into semantic groups for training diagnostics.

    :param net: Network whose linear pathway, latent projection, hidden layers, normalization
        layers, noise transform, and output layer are inspected.
    :type net: _EnSurvNet

    :returns: Group labels mapped to names used by ``net.named_parameters()``.
    :rtype: dict[str, list[str]]
    """
    groups = {}
    if net.apply_linear:
        groups['beta (linear)'] = ['beta']
    if net.W is not None:
        groups['W (latent proj)'] = ['W']
    for i in range(net.num_layer):
        keys = [f'layers.{i}.weight', f'layers.{i}.bias']
        if net.bns is not None:
            keys += [f'bns.{i}.weight', f'bns.{i}.bias']
        groups[f'layer {i}'] = keys
    if net.noise_transform is not None:
        groups['noise transform'] = [name for name, _ in net.named_parameters() if name.startswith('noise_transform.')]
    groups['final_layer'] = ['final_layer.weight', 'final_layer.bias']
    return groups


def _group_grad_norms(net, param_groups):
    r"""Compute the root mean square of available gradients in each parameter group.

    :param net: Network with gradients populated by the current backward pass.
    :type net: _EnSurvNet
    :param param_groups: Semantic group labels and corresponding named parameters, as returned
        by ``_get_param_groups``.
    :type param_groups: dict[str, list[str]]

    :returns: One RMS gradient magnitude per group; groups without gradients receive zero.
    :rtype: dict[str, float]

    Call after ``loss.backward()`` and before clearing gradients. For a group with
    :math:`N_g` available gradient entries, the reported value is
    :math:`\{\sum_{j=1}^{N_g} (\nabla_j L)^2 / \max(N_g,1)\}^{1/2}`.
    """
    named = dict(net.named_parameters())
    out = {}
    for gname, keys in param_groups.items():
        sq, cnt = 0.0, 0
        for k in keys:
            p = named.get(k, None)
            if p is not None and p.grad is not None:
                sq += p.grad.detach().pow(2).sum().item()
                cnt += p.grad.numel()
        out[gname] = (sq / max(cnt, 1)) ** 0.5
    return out


def _extract_raw_importance(net, input_dim):
    r"""Extract descriptive linear coefficients and nonlinear structural norms.

    :param net: Network whose current parameter values are inspected without a forward pass.
    :type net: _EnSurvNet
    :param input_dim: Number of covariates; used to select feature columns from the first
        hidden-layer weight matrix.
    :type input_dim: int

    :returns: Linear coefficients, or None when disabled, and one nonlinear norm per feature.
    :rtype: tuple[numpy.ndarray or None, numpy.ndarray]

    With a latent projection, use row norms of its weight matrix. Otherwise multiply compatible
    absolute hidden-layer weight matrices and take feature-column norms. 
    """
    beta_weights = None
    if net.apply_linear:
        beta_weights = net.beta.detach().cpu().numpy()
 
    if hasattr(net, 'W') and net.W is not None:
        W_weights = net.W.detach().cpu().numpy()
        nonlinear_importance = np.linalg.norm(W_weights, axis=1)
    else:
        path_matrix = None
        for layer in net.layers:
            if isinstance(layer, nn.Linear):
                w = np.abs(layer.weight.detach().cpu().numpy())
                if path_matrix is None:
                    path_matrix = w[:, :input_dim]
                else:
                    if w.shape[1] == path_matrix.shape[0]:
                        path_matrix = w @ path_matrix
                    else:
                        break
        nonlinear_importance = np.linalg.norm(path_matrix, axis=0)
 
    return beta_weights, nonlinear_importance

