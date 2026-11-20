"""EnSurv neural generator and estimator."""

import copy
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from ._random import _prediction_rng
from ._diagnostics import _get_param_groups, _group_grad_norms, _extract_raw_importance
from .horizon import _validate_outcomes, resolve_tau, apply_tau_restriction
from .km import kaplan_meier_batched
from .loss import survival_engression_loss


def get_act_func(name):
    r"""Construct the requested hidden-layer activation.

    :param name: Activation name: 'relu', 'sigmoid', 'tanh', 'softmax', 'elu', or 'softplus'.
    :type name: str

    :returns: Activation instance, or None for an unsupported name. Softmax uses dimension 1;
        ReLU and ELU operate in place.
    :rtype: torch.nn.Module or None
    """
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "sigmoid":
        return nn.Sigmoid() 
    elif name == "tanh":
        return nn.Tanh() 
    elif name == "softmax":
        return nn.Softmax(dim=1)
    elif name == "elu":
        return nn.ELU(inplace=True)
    elif name == "softplus":
        return nn.Softplus()
    else:
        return None


class _EnSurvNet(nn.Module):
    r"""Neural generator for patient-specific log-time draws.

    :param input_dim: Number of input covariates.
    :type input_dim: int
    :param noise_dim: Dimension of each independent simulation-noise vector; default 64.
    :type noise_dim: int, optional
    :param num_layer: Number of hidden fully connected layers; default 1.
    :type num_layer: int, optional
    :param hidden_dims: One width per hidden layer. None selects [32].
    :type hidden_dims: list[int] or None, optional
    :param hidden_act: Hidden activation name accepted by get_act_func; default relu.
    :type hidden_act: str, optional
    :param dropout: Dropout probability after each hidden activation. Zero disables dropout.
    :type dropout: float, optional
    :param latent_dim: Dimension of a learned covariate projection. None retains the input
        feature dimension.
    :type latent_dim: int or None, optional
    :param apply_linear: Whether to add a direct linear covariate pathway, initialized with zero
        coefficients.
    :type apply_linear: bool, optional
    :param batch_norm: Whether to normalize each hidden-layer output before activation; default
        False.
    :type batch_norm: bool, optional
    :param noise_all_layer: In concatenated mode, append the same raw noise after each hidden
        layer, including before the final layer. Pre-additive mode does not use this option.
    :type noise_all_layer: bool, optional
    :param generator: Noise-injection formulation. Pre-additive transforms noise into input
        space; concatenated appends raw noise to projected or original features.
    :type generator: {'pre_additive', 'concatenated'}, optional

    The default pre-additive generator is

    .. math::

       U_\nu(x,\epsilon)=g_\theta\{x+h_\phi(\epsilon)\}+\beta^\top x.

    A latent projection, when enabled, is applied after adding transformed noise.
    Concatenated mode retains the alternative architecture used by the notebook.
    Outputs are scalar raw log-times and are not capped at the restriction horizon.
    """
    def __init__(self, input_dim, noise_dim=64, num_layer=1, hidden_dims=None, hidden_act="relu", dropout=0.0, latent_dim=None, 
                 apply_linear=True, batch_norm=False, noise_all_layer=True, generator="pre_additive"):
        r"""Initialize generator layers and their original weight initializations.

        :param input_dim: Number of input covariates.
        :type input_dim: int
        :param noise_dim: Dimension of each independent simulation-noise vector; default 64.
        :type noise_dim: int, optional
        :param num_layer: Number of hidden fully connected layers; default 1.
        :type num_layer: int, optional
        :param hidden_dims: One width per hidden layer. None selects [32].
        :type hidden_dims: list[int] or None, optional
        :param hidden_act: Hidden activation name accepted by get_act_func; default relu.
        :type hidden_act: str, optional
        :param dropout: Dropout probability after each hidden activation. Zero disables dropout.
        :type dropout: float, optional
        :param latent_dim: Dimension of a learned covariate projection. None retains the input
            feature dimension.
        :type latent_dim: int or None, optional
        :param apply_linear: Whether to add a direct linear covariate pathway, initialized with zero
            coefficients.
        :type apply_linear: bool, optional
        :param batch_norm: Whether to normalize each hidden-layer output before activation; default
            False.
        :type batch_norm: bool, optional
        :param noise_all_layer: In concatenated mode, append the same raw noise after each hidden
            layer, including before the final layer. Pre-additive mode does not use this option.
        :type noise_all_layer: bool, optional
        :param generator: Noise-injection formulation. Pre-additive transforms noise into input
            space; concatenated appends raw noise to projected or original features.
        :type generator: {'pre_additive', 'concatenated'}, optional

        The default pre-additive generator is

        .. math::

           U_\nu(x,\epsilon)=g_\theta\{x+h_\phi(\epsilon)\}+\beta^\top x.

        A latent projection, when enabled, is applied after adding transformed noise.
        Concatenated mode retains the alternative architecture used by the notebook.
        Outputs are scalar raw log-times and are not capped at the restriction horizon.
        """
        super(_EnSurvNet, self).__init__()
        if hidden_dims is None:
            hidden_dims = [32]
            
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.num_layer = num_layer
        self.noise_all_layer = noise_all_layer
        self.apply_linear = apply_linear
        self.generator = generator
        self.noise_transform = (nn.Sequential(nn.Linear(noise_dim, input_dim), nn.Tanh(),
                                             nn.Linear(input_dim, input_dim))
                                if generator == 'pre_additive' else None)
        
        if latent_dim is not None:
            self.W = nn.Parameter(torch.empty(input_dim, latent_dim))
            nn.init.xavier_uniform_(self.W)
            feature_dim = latent_dim
        else:
            self.W = None
            feature_dim = input_dim

        if self.apply_linear:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if batch_norm else None
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList() if dropout > 0.0 else None
        
        # Pre-additive mode uses the feature dimension; legacy mode concatenates noise.
        current_dim = feature_dim if generator == 'pre_additive' else feature_dim + noise_dim
        
        for i in range(num_layer):
            self.layers.append(nn.Linear(current_dim, hidden_dims[i]))
            
            if batch_norm:
                self.bns.append(nn.BatchNorm1d(hidden_dims[i]))
                
            self.activations.append(get_act_func(hidden_act))
            
            if dropout > 0.0:
                self.dropouts.append(nn.Dropout(dropout))
                
            # Update current_dim for the next layer
            current_dim = hidden_dims[i]
            # If injecting noise at every layer, add noise_dim to the expected input of the next layer
            if self.noise_all_layer and self.generator == 'concatenated':
                current_dim += noise_dim
                
        # Final linear layer to output the single scalar value
        self.final_layer = nn.Linear(current_dim, 1)
        
        # Apply Xavier Glorot initialization
        for m in self.layers:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
        nn.init.xavier_uniform_(self.final_layer.weight)
        if self.final_layer.bias is not None:
            nn.init.zeros_(self.final_layer.bias)

    def forward(self, X, noise):
        r"""Generate raw log-time draws for a batch of covariates and noise.

        :param X: Covariates on the network device, normally float32.
        :type X: torch.Tensor of shape (n, input_dim)
        :param noise: Simulation-noise draws on the same device. Each observation receives m draws.
        :type noise: torch.Tensor of shape (n, m, noise_dim)

        :returns: Nonlinear log-time draws plus the direct linear pathway when enabled.
        :rtype: torch.Tensor of shape (n, m)
        """
        n, m, _ = noise.shape
        if self.generator == 'pre_additive':
            noisy_x = X[:, None, :] + self.noise_transform(noise)
            features = noisy_x @ self.W if self.W is not None else noisy_x
            current_state = features.reshape(n * m, -1)
        else:
            features = X @ self.W if self.W is not None else X
            features = features[:, None, :].expand(-1, m, -1)
            current_state = torch.cat((features.reshape(n * m, -1), noise.reshape(n * m, -1)), -1)
        noise_2d = noise.reshape(n * m, -1)

        for i in range(self.num_layer):
            current_state = self.layers[i](current_state)
            
            if self.bns is not None:
                current_state = self.bns[i](current_state)
                
            current_state = self.activations[i](current_state)
            
            if self.dropouts is not None:
                current_state = self.dropouts[i](current_state)
                
            # If noise_all_layer is True, append the raw noise back onto the hidden state
            # before it passes to the next iteration or the final layer
            if self.noise_all_layer and self.generator == 'concatenated':
                current_state = torch.cat([current_state, noise_2d], dim=-1)
                
        nonlinear_2d = self.final_layer(current_state)
        
        # Reshape back to 3D and project
        nonlinear = nonlinear_2d.view(n, m, -1).squeeze(-1)

        linear = torch.matmul(X, self.beta).unsqueeze(1) if self.apply_linear else 0
        
        return nonlinear + linear


class EnSurv:
    r"""Fit and sample an EnSurv pooled-completion survival generator.

    :param latent_dim: Learned projection dimension; None leaves covariates unprojected.
    :type latent_dim: int or None, optional
    :param num_layer: Number of hidden layers, at least one; default 1.
    :type num_layer: int, optional
    :param hidden_dims: One hidden width per layer; the original default is [32].
    :type hidden_dims: list[int], optional
    :param dropout: Hidden-layer dropout probability; default zero.
    :type dropout: float, optional
    :param hidden_act: Hidden activation: relu, sigmoid, tanh, softmax, elu, or softplus.
        Support is checked during fit.
    :type hidden_act: str, optional
    :param apply_linear: Whether the generator has a direct linear covariate pathway; default
        True.
    :type apply_linear: bool, optional
    :param batch_normalization: Enable hidden-layer batch normalization; default False.
    :type batch_normalization: bool, optional
    :param noise_all_layer: Append raw noise after every hidden layer in concatenated mode;
        ignored in pre-additive mode.
    :type noise_all_layer: bool, optional
    :param noise_dim: Positive simulation-noise dimension, only used when generator is 'concatenated'; default 64.
    :type noise_dim: int, optional
    :param noise_dist: Uniform draws use the interval [-noise_std, noise_std]. Gaussian draws
        have zero mean and standard deviation noise_std.
    :type noise_dist: {'uniform', 'gaussian'}, optional
    :param noise_std: Positive finite noise scale; a half-width in uniform mode and a standard
        deviation in Gaussian mode.
    :type noise_std: float, optional
    :param device: Training and prediction device. None selects CUDA if available, otherwise
        CPU.
    :type device: torch.device, str or None, optional
    :param random_state: Seed reset at the start of each fit for NumPy and PyTorch. None retains
        current random states.
    :type random_state: int or None, optional
    :param tau: Default restriction horizon in input Y units. 'max_event' uses the largest
        observed failure in the supplied training data.
    :type tau: float or {'max_event'}, optional
    :param time_scale: Positive time-unit divisor used internally during fit. Predictions are
        converted back into fit-input units; default 1.0.
    :type time_scale: float, optional
    :param generator: Generator architecture; default pre_additive. The supplied CV notebook
        explicitly selects concatenated.
    :type generator: {'pre_additive', 'concatenated'}, optional

    The fitted generator learns a conditional approximation to the pooled-KM
    pseudo-complete outcome law. Numeric :math:`\tau` defines a fixed horizon;
    ``tau='max_event'`` defines a training-sample-dependent horizon.

    Fitted attributes include ``model``, ``input_dim``, ``tau_``, ``tau_source_``,
    ``time_scale_``, ``log_tau_``, ``noise_mode_``, ``original_event_count_``,
    ``followup_at_tau_count_``, ``restricted_complete_count_``, and
    ``early_censored_count_``. ``history`` stores training diagnostics when enabled.
    Raw outputs remain uncapped; extrapolation beyond the horizon is a property
    of the working generator rather than an identified survival estimate.
    """
    def __init__(self, latent_dim=None, num_layer=1, hidden_dims=[32], dropout=0.0, hidden_act="relu", apply_linear=True,
                 batch_normalization=False, noise_all_layer=True, noise_dim=64, noise_dist='uniform',
                 noise_std=1.0, device=None, random_state=None, tau='max_event',
                 time_scale=1.0, generator='pre_additive'):
        r"""Configure the generator, noise distribution, horizon, and time units.

        :param latent_dim: Learned projection dimension; None leaves covariates unprojected.
        :type latent_dim: int or None, optional
        :param num_layer: Number of hidden layers, at least one; default 1.
        :type num_layer: int, optional
        :param hidden_dims: One hidden width per layer; the original default is [32].
        :type hidden_dims: list[int], optional
        :param dropout: Hidden-layer dropout probability; default zero.
        :type dropout: float, optional
        :param hidden_act: Hidden activation: relu, sigmoid, tanh, softmax, elu, or softplus.
            Support is checked during fit.
        :type hidden_act: str, optional
        :param apply_linear: Whether the generator has a direct linear covariate pathway; default
            True.
        :type apply_linear: bool, optional
        :param batch_normalization: Enable hidden-layer batch normalization; default False.
        :type batch_normalization: bool, optional
        :param noise_all_layer: Append raw noise after every hidden layer in concatenated mode;
            ignored in pre-additive mode.
        :type noise_all_layer: bool, optional
        :param noise_dim: Positive simulation-noise dimension; default 64.
        :type noise_dim: int, optional
        :param noise_dist: Uniform draws use the interval [-noise_std, noise_std]. Gaussian draws
            have zero mean and standard deviation noise_std.
        :type noise_dist: {'uniform', 'gaussian'}, optional
        :param noise_std: Positive finite noise scale; a half-width in uniform mode and a standard
            deviation in Gaussian mode.
        :type noise_std: float, optional
        :param device: Training and prediction device. None selects CUDA if available, otherwise
            CPU.
        :type device: torch.device, str or None, optional
        :param random_state: Seed reset at the start of each fit for NumPy and PyTorch. None retains
            current random states.
        :type random_state: int or None, optional
        :param tau: Default restriction horizon in input Y units. 'max_event' uses the largest
            observed failure in the supplied training data.
        :type tau: float or {'max_event'}, optional
        :param time_scale: Positive time-unit divisor used internally during fit. Predictions are
            converted back into fit-input units; default 1.0.
        :type time_scale: float, optional
        :param generator: Generator architecture; default pre_additive. The supplied CV notebook
            explicitly selects concatenated.
        :type generator: {'pre_additive', 'concatenated'}, optional

        The fitted generator learns a conditional approximation to the pooled-KM
        pseudo-complete outcome law. Numeric :math:`\tau` defines a fixed horizon;
        ``tau='max_event'`` defines a training-sample-dependent horizon.

        Fitted attributes include ``model``, ``input_dim``, ``tau_``, ``tau_source_``,
        ``time_scale_``, ``log_tau_``, ``noise_mode_``, ``original_event_count_``,
        ``followup_at_tau_count_``, ``restricted_complete_count_``, and
        ``early_censored_count_``. ``history`` stores training diagnostics when enabled.
        Raw outputs remain uncapped; extrapolation beyond the horizon is a property
        of the working generator rather than an identified survival estimate.
        """
        if generator not in ('pre_additive', 'concatenated'):
            raise ValueError("generator must be 'pre_additive' or 'concatenated'.")
        if num_layer < 1 or len(hidden_dims) != num_layer:
            raise ValueError('hidden_dims must provide one width per layer, and num_layer>=1.')
        if noise_dim < 1 or noise_std <= 0 or not np.isfinite(noise_std):
            raise ValueError('noise_dim and noise_std must be positive.')
        if noise_dist not in ['uniform', 'gaussian']:
            raise ValueError("noise_dist must be 'uniform' or 'gaussian'")
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = None
        self.latent_dim = latent_dim
        self.num_layer = num_layer
        self.hidden_dims = hidden_dims
        self.noise_dim = noise_dim
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.batch_normalization = batch_normalization
        self.noise_all_layer = noise_all_layer
        self.noise_dist = noise_dist
        self.noise_std = noise_std
        self.apply_linear = apply_linear
        self.generator = generator
        self.device = device
        self.random_state = random_state
        self.model = None
        self.tau = tau
        self.time_scale = time_scale
        
    def _set_seed(self):
        r"""Reset NumPy and PyTorch seeds when random_state is configured.

        Uses ``self.random_state`` for NumPy, PyTorch CPU, and the current CUDA device when
        available. If it is None, no seed is set.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            
    def _sample_noise(self, n, m, dim):
        r"""Draw simulation noise from the configured distribution.

        :param n: Number of observations in the batch.
        :type n: int
        :param m: Number of independent draws per observation.
        :type m: int
        :param dim: Number of components per noise vector.
        :type dim: int

        :returns: Noise on self.device with scale self.noise_std.
        :rtype: torch.Tensor of shape (n, m, dim)

        Uniform components are sampled on :math:`[-s,s]`; Gaussian components have law
        :math:`\mathcal N(0,s^2)`, where :math:`s=\mathrm{noise\_std}`.
        """
        if self.noise_dist == 'gaussian':
            return torch.randn(n, m, dim, device=self.device) * self.noise_std
        return (torch.rand(n, m, dim, device=self.device) * 2 - 1) * self.noise_std

    def fit(self, X, Y, Delta, batch_size=None, lr=1e-3, epochs=100, m=10, B=50, lambda_energy=0.5, 
            verbose=False, patience=None, weight_decay=None, l1_reg=0.0, track_diagnostics=True, 
            diag_every=1, feature_names=None, domain_constraints=None, lambda_constraint=1.0,
            tau=None, time_scale=None, km_batch_size=256, noise_mode="fixed"):
        r"""Train raw generator outputs against horizon-restricted and pooled-completed outcomes.

        :param X: Nonempty finite covariate matrix converted to float32. This method does not impute
            or standardize features.
        :type X: array-like of shape (n, p)
        :param Y: Original positive observed follow-up times in the desired input time units.
        :type Y: array-like of shape (n,)
        :param Delta: Original binary failure indicators: 1 for failure and 0 for censoring.
        :type Delta: array-like of shape (n,)
        :param batch_size: Training observations per optimizer step. None uses the full training
            sample.
        :type batch_size: int or None, optional
        :param lr: Adam learning rate; default 1e-3.
        :type lr: float, optional
        :param epochs: Maximum training epochs, at least one; default 100.
        :type epochs: int, optional
        :param m: Simulation draws per observation in the training score, at least two; default 10.
        :type m: int, optional
        :param B: Number of n-observation bootstrap samples for pooled KM completion, at least one;
            default 50.
        :type B: int, optional
        :param lambda_energy: Weight subtracted times the pairwise energy term; default 0.5. Other
            weights define another scoring objective.
        :type lambda_energy: float, optional
        :param verbose: Show the training progress bar and early-stopping message.
        :type verbose: bool, optional
        :param patience: Consecutive epochs without a strictly smaller training loss before
            stopping. None disables early stopping; when enabled, the best observed weights are
            restored.
        :type patience: int or None, optional
        :param weight_decay: Weight decay passed to Adam. None leaves the optimizer default
            unchanged.
        :type weight_decay: float or None, optional
        :param l1_reg: Nonnegative coefficient for the L1 norms of the linear coefficients and
            latent projection, or the first-layer feature weights without a projection. Default
            zero.
        :type l1_reg: float, optional
        :param track_diagnostics: Record loss components, parameter-group gradient RMS values, and
            structural feature-importance snapshots in history.
        :type track_diagnostics: bool, optional
        :param diag_every: Epoch interval for importance and weight-norm snapshots, at least one.
            The final epoch is also recorded; per-epoch loss and gradient diagnostics are retained
            separately.
        :type diag_every: int, optional
        :param feature_names: One name per X column, used for diagnostics and mapping named domain
            constraints.
        :type feature_names: sequence[str] or None, optional
        :param domain_constraints: Features assigned to 'positive' or 'negative' derivative
            directions. Names require feature_names; integer indices address columns directly.
        :type domain_constraints: dict[str, sequence[int or str]] or None, optional
        :param lambda_constraint: Nonnegative coefficient for penalizing derivatives of the
            draw-averaged raw log-time that violate the specified directions; default 1.0.
        :type lambda_constraint: float, optional
        :param tau: Horizon override in Y units. None uses the constructor setting; the
            maximum-event choice uses training failures only.
        :type tau: float, {'max_event'} or None, optional
        :param time_scale: Positive finite divisor for internal times, overriding the constructor
            when supplied. None uses self.time_scale.
        :type time_scale: float or None, optional
        :param km_batch_size: Bootstrap resamples processed together during KM estimation; default
            256. This is separate from the optimizer batch size.
        :type km_batch_size: int, optional
        :param noise_mode: Fixed mode samples one observation-specific noise bank on CPU and reuses
            it across epochs. Resample mode draws fresh noise for each batch.
        :type noise_mode: {'fixed', 'resample'}, optional

        :returns: The fitted instance, enabling chained calls.
        :rtype: EnSurv

        :raises ValueError: Inputs or checked configuration values violate the method requirements.

        Training uses original indicators to estimate pooled KM curves, then constructs
        :math:`Y_\tau` and :math:`D` for the score. The generated coordinate is
        :math:`U=\log(T/\mathrm{time\_scale})`; the horizon in this coordinate is
        :math:`\log(\tau/\mathrm{time\_scale})`.

        Domain constraints penalize sign violations in the average generated log-time
        gradient and do not enforce a hard monotonicity constraint. Diagnostics retain
        the full recorded training path even when early stopping restores earlier
        weights. Fixed-noise, fixed-horizon unpenalized theory does not automatically
        extend to dropout, batch normalization, resampled noise, or nonvanishing penalties.
        """
        if not np.isfinite(l1_reg) or l1_reg < 0 or not np.isfinite(lambda_constraint) or lambda_constraint < 0:
            raise ValueError('Regularization and domain-penalty weights must be finite and nonnegative.')
        if get_act_func(self.hidden_act) is None:
            raise ValueError('Unsupported hidden activation.')
        self._set_seed()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or len(X) == 0 or not np.isfinite(X).all():
            raise ValueError('X must be a nonempty finite two-dimensional matrix.')
        original_y, original_delta = _validate_outcomes(Y, Delta, device=self.device)
        n, p = X.shape
        if len(original_y) != n:
            raise ValueError('X and outcomes have different sample sizes.')
        if m < 2 or B < 1 or epochs < 1 or diag_every < 1:
            raise ValueError('Require m>=2, B>=1, epochs>=1 and diag_every>=1.')
        # if lambda_energy != 0.5:
        #     raise ValueError('The theoretical energy coefficient is 0.5.')
        if noise_mode not in ('fixed', 'resample'):
            raise ValueError("noise_mode must be 'fixed' or 'resample'.")
        if feature_names is not None and len(feature_names) != p:
            raise ValueError('feature_names must have one entry per input feature.')
        scale = float(self.time_scale if time_scale is None else time_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('time_scale must be a finite positive fixed unit conversion.')
        requested_tau = self.tau if tau is None else tau
        self.tau_, self.tau_source_ = resolve_tau(original_y, original_delta, requested_tau)
        self.time_scale_ = scale
        self.log_tau_ = np.log(self.tau_ / scale)
        self.input_dim = p
        self.original_event_count_ = int(original_delta.sum().item())
        self.followup_at_tau_count_ = int((original_y >= self.tau_).sum().item())
        self.noise_mode_ = noise_mode
        if batch_size is None:
            batch_size = n
        if batch_size < 1:
            raise ValueError('batch_size must be positive.')

        # Process Domain Constraints (Map Strings to Indices)
        internal_constraints = None
        if domain_constraints is not None:
            internal_constraints = {}
            # Check if any constraint item is a string
            has_strings = any(isinstance(item, str) for items in domain_constraints.values() for item in items)
            
            if has_strings:
                if feature_names is None:
                    raise ValueError("feature_names must be provided when domain_constraints uses string feature names.")
                
                name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
                
                for direction, items in domain_constraints.items():
                    internal_constraints[direction] = []
                    for item in items:
                        if isinstance(item, str):
                            if item not in name_to_idx:
                                raise ValueError(f"Constraint feature '{item}' not found in feature_names.")
                            internal_constraints[direction].append(name_to_idx[item])
                        else:
                            internal_constraints[direction].append(item) # Fallback if integers are mixed
            else:
                internal_constraints = domain_constraints
        if internal_constraints is not None:
            if set(internal_constraints) - {'positive', 'negative'}:
                raise ValueError("Constraint directions must be 'positive' or 'negative'.")
            for indices_for_direction in internal_constraints.values():
                if any(not isinstance(k, (int, np.integer)) or k < 0 or k >= p for k in indices_for_direction):
                    raise ValueError('Constraint feature indices must be in [0,input_dim).')
            
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y_original_scaled = original_y / self.time_scale_
        tau_scaled = self.tau_ / self.time_scale_
        Y_t, D_t = apply_tau_restriction(Y_original_scaled, original_delta, tau_scaled)
        self.restricted_complete_count_ = int(D_t.sum().item())
        self.early_censored_count_ = int((D_t == 0).sum().item())
        
        self.model = _EnSurvNet(
            input_dim=p, 
            noise_dim=self.noise_dim, 
            num_layer=self.num_layer, 
            hidden_dims=self.hidden_dims, 
            hidden_act=self.hidden_act, 
            dropout=self.dropout, 
            latent_dim=self.latent_dim, 
            batch_norm=self.batch_normalization,
            noise_all_layer=self.noise_all_layer, generator=self.generator
        ).to(self.device)
        
        if weight_decay is not None:
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
        indices = np.random.randint(0, n, size=(B, n))
        km_bootstrap = kaplan_meier_batched(Y_original_scaled, original_delta, indices,
                                           close_tail=True, device=self.device,
                                           tau=tau_scaled, bootstrap_batch_size=km_batch_size)
        # Stored on CPU so the fixed bank need not occupy GPU memory during fitting.
        fixed_noise = None
        if noise_mode == 'fixed':
            if self.noise_dist == 'gaussian':
                fixed_noise = torch.randn(n, m, self.noise_dim) * self.noise_std
            else:
                fixed_noise = (torch.rand(n, m, self.noise_dim) * 2 - 1) * self.noise_std

        if track_diagnostics:
            param_groups = _get_param_groups(self.model)
            self.history = {
                'epoch': [],
                'total_loss': [],
                'event_loss': [],
                'censor_loss': [],
                'energy_term': [],
                'constraint_loss': [], 
                'lambda_energy': lambda_energy,
                'tau': self.tau_, 'tau_source': self.tau_source_,
                'time_scale': self.time_scale_, 'noise_mode': noise_mode,
                'restricted_complete_count': self.restricted_complete_count_,
                'early_censored_count': self.early_censored_count_,
                'grad_norms': {g: [] for g in param_groups},
                'importance_epochs': [],
                'importance_pct': [],
                'linear_beta_raw': [] if self.apply_linear else None,
                'nonlinear_importance_raw': [],
                'weight_norms': {'beta_l1': [] if self.apply_linear else None, 'nonlinear_total': []},
                'feature_names': feature_names,
            }
        else:
            self.history = None
            param_groups = None
            
        self.model.train()
        pbar = tqdm(range(epochs), desc="Training", disable=not verbose)

        best_loss = float('inf')
        epochs_no_improve = 0
        best_model_wts = None

        for epoch in pbar:
            permutation = torch.randperm(n, device=self.device)
            epoch_loss = 0.0

            if track_diagnostics:
                epoch_event_loss = 0.0
                epoch_censor_loss = 0.0
                epoch_energy = 0.0
                epoch_constraint = 0.0 
                epoch_grad_accum = {g: 0.0 for g in param_groups}
                n_batches = 0

            for i in range(0, n, batch_size):
                batch_indices = permutation[i:i+batch_size]

                X_batch = X_t[batch_indices].clone().detach()
                if internal_constraints and lambda_constraint > 0:
                    X_batch.requires_grad_(True)
                    
                Y_batch = Y_t[batch_indices]
                D_batch = D_t[batch_indices]

                optimizer.zero_grad()

                current_b_size = X_batch.size(0)
                noise = (fixed_noise[batch_indices.cpu()].to(self.device) if fixed_noise is not None
                         else self._sample_noise(current_b_size, m, self.noise_dim))

                logT_hat = self.model(X_batch, noise)

                if track_diagnostics:
                    loss, comps = survival_engression_loss(
                        logT_hat, Y_batch, D_batch, km_bootstrap, lambda_energy,
                        return_components=True)
                else:
                    loss = survival_engression_loss(logT_hat, Y_batch, D_batch, km_bootstrap, lambda_energy)

                if l1_reg > 0.0:
                    l1_penalty = (torch.norm(self.model.beta, 1) if self.apply_linear
                                  else X_batch.new_zeros(()))

                    if self.model.W is not None:
                        l1_penalty = l1_penalty + torch.norm(self.model.W, 1)
                    else:
                        l1_penalty = l1_penalty + torch.norm(self.model.layers[0].weight[:, :self.input_dim], 1)

                    loss = loss + l1_reg * l1_penalty

                # Domain constraint penalty
                constraint_penalty_val = 0.0
                if internal_constraints and lambda_constraint > 0:
                    grad_outputs = torch.ones_like(logT_hat)
                    grad_X = torch.autograd.grad(
                        outputs=logT_hat,
                        inputs=X_batch,
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True
                    )[0] / m  

                    constraint_penalty = torch.tensor(0.0, device=self.device)
                    
                    if internal_constraints.get('positive'):
                        pos_indices = internal_constraints['positive']
                        violating_grads = torch.nn.functional.relu(-grad_X[:, pos_indices])
                        constraint_penalty += violating_grads.sum(dim=1).mean()
                        
                    if internal_constraints.get('negative'):
                        neg_indices = internal_constraints['negative']
                        violating_grads = torch.nn.functional.relu(grad_X[:, neg_indices])
                        constraint_penalty += violating_grads.sum(dim=1).mean()

                    loss = loss + lambda_constraint * constraint_penalty
                    constraint_penalty_val = constraint_penalty.item() * current_b_size

                loss.backward()

                if track_diagnostics:
                    gnorms = _group_grad_norms(self.model, param_groups)
                    for g in param_groups:
                        epoch_grad_accum[g] += gnorms[g]
                    epoch_event_loss += comps['event_loss'].item() * current_b_size
                    epoch_censor_loss += comps['censor_loss'].item() * current_b_size
                    epoch_energy += comps['energy_term'].item() * current_b_size
                    epoch_constraint += constraint_penalty_val 
                    n_batches += 1

                optimizer.step()
                epoch_loss += loss.item() * current_b_size

            epoch_loss /= n

            if track_diagnostics:
                self.history['epoch'].append(epoch)
                self.history['total_loss'].append(epoch_loss)
                self.history['event_loss'].append(epoch_event_loss / n)
                self.history['censor_loss'].append(epoch_censor_loss / n)
                self.history['energy_term'].append(epoch_energy / n)
                self.history['constraint_loss'].append(epoch_constraint / n) 
                for g in param_groups:
                    self.history['grad_norms'][g].append(epoch_grad_accum[g] / max(n_batches, 1))

                if (epoch % diag_every == 0) or (epoch == epochs - 1):
                    beta_w, nonlin_w = _extract_raw_importance(self.model, self.input_dim)
                    total_raw = (np.abs(beta_w) + nonlin_w) if self.apply_linear else nonlin_w
                    pct = (total_raw / total_raw.sum() * 100) if total_raw.sum() > 0 else np.zeros_like(total_raw)

                    self.history['importance_epochs'].append(epoch)
                    self.history['importance_pct'].append(pct)
                    self.history['nonlinear_importance_raw'].append(nonlin_w)
                    self.history['weight_norms']['nonlinear_total'].append(float(nonlin_w.sum()))
                    if self.apply_linear:
                        self.history['linear_beta_raw'].append(beta_w)
                        self.history['weight_norms']['beta_l1'].append(float(np.abs(beta_w).sum()))

            if verbose:
                pbar.set_postfix(loss=f"{epoch_loss:.4f}")

            if patience is not None:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                    break

        if track_diagnostics:
            self.history['importance_pct'] = np.array(self.history['importance_pct'])
            self.history['nonlinear_importance_raw'] = np.array(self.history['nonlinear_importance_raw'])
            if self.apply_linear:
                self.history['linear_beta_raw'] = np.array(self.history['linear_beta_raw'])

        if patience is not None and best_model_wts is not None:
            self.model.load_state_dict(best_model_wts)

        return self


    def explain(self, feature_names=None, y_max=None):
        r"""Summarize fitted pathway coefficients and structural feature-weight norms.

        :param feature_names: Names matching the fitted input columns. None generates Feature_0,
            Feature_1, and so on.
        :type feature_names: sequence[str] or None, optional
        :param y_max: Deprecated compatibility argument. It does not rescale effects; supplying it
            with a linear pathway emits a DeprecationWarning.
        :type y_max: float or None, optional

        :returns: Features sorted by Total_Importance_Pct, with NonLinear_Contribution_Norm and,
            when enabled, Linear_Beta_Raw.
        :rtype: pandas.DataFrame

        :raises ValueError: The model has not been fitted.

        Importance percentages normalize :math:`|\beta_j|+r_j`, or :math:`r_j` without
        a linear pathway, to sum to 100 when the total is positive. Nonlinear norms
        use latent-projection rows or compatible absolute hidden-layer path weights.
        These quantities are descriptive network summaries providing global explainability.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
            
        self.model.eval()

        if self.apply_linear:
            beta_weights = self.model.beta.detach().cpu().numpy()
        
        # Extract Non-Linear Contribution (Path-Weight through ALL Linear layers)
        if hasattr(self.model, 'W') and self.model.W is not None:
            W_weights = self.model.W.detach().cpu().numpy()  
            nonlinear_importance = np.linalg.norm(W_weights, axis=1)
        else:
            path_matrix = None
            
            for layer in self.model.layers:
                # Only target fully connected linear layers, ignore BatchNorm/LayerNorm
                if isinstance(layer, nn.Linear):
                    w = np.abs(layer.weight.detach().cpu().numpy()) 
                    
                    if path_matrix is None:
                        # First layer: slice out the noise dimensions
                        path_matrix = w[:, :self.input_dim] 
                    else:
                        # Safety check: multiply only if dimensions align sequentially
                        if w.shape[1] == path_matrix.shape[0]:
                            path_matrix = w @ path_matrix
                        else:
                            # If the model uses skip-connections (w.shape[1] > path_matrix.shape[0]),
                            # tracking intrinsic paths via simple dot products breaks. We break the 
                            # loop and accept the structural path calculated up to this point.
                            break
            
            nonlinear_importance = np.linalg.norm(path_matrix, axis=0)
            
        # Calculate Total Importance (Raw network capacity allocation)
        if self.apply_linear:
            total_raw_importance = np.abs(beta_weights) + nonlinear_importance
        else:
            total_raw_importance = nonlinear_importance
            
        # Normalize to 100%
        if total_raw_importance.sum() > 0:
            importance_pct = (total_raw_importance / total_raw_importance.sum()) * 100
        else:
            importance_pct = np.zeros_like(total_raw_importance)
            
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(self.input_dim)]

        # 4. Construct DataFrame
        df_dict = {
            'Feature': feature_names,
            'Total_Importance_Pct': importance_pct,
        }

        if self.apply_linear:
            df_dict['Linear_Beta_Raw'] = beta_weights
            if y_max is not None:
                warnings.warn('y_max is deprecated in explain(): log-time pathway coefficients are not additive day-scale effects.',
                              DeprecationWarning, stacklevel=2)
                
        df_dict['NonLinear_Contribution_Norm'] = nonlinear_importance
        
        df = pd.DataFrame(df_dict)
        df = df.sort_values(by='Total_Importance_Pct', ascending=False).reset_index(drop=True)
        
        return df

    def _prediction_inputs(self, X, nsample, batch_size):
        r"""Validate prediction inputs against the fitted feature dimension.

        :param X: Finite prediction covariates; an empty row dimension is allowed.
        :type X: array-like of shape (n, input_dim)
        :param nsample: Positive number of simulation draws per observation.
        :type nsample: int
        :param batch_size: Positive prediction batch size.
        :type batch_size: int

        :returns: Validated covariates converted to float32.
        :rtype: numpy.ndarray

        :raises RuntimeError: The model has not been fitted.
        :raises ValueError: The feature matrix or sampling configuration is invalid.
        """
        if self.model is None:
            raise RuntimeError('Fit the model before predicting.')
        x = np.asarray(X, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.input_dim or not np.isfinite(x).all():
            raise ValueError('X must be a finite matrix with the fitted feature dimension.')
        if not isinstance(nsample, (int, np.integer)) or nsample < 1 or batch_size < 1:
            raise ValueError('nsample and batch_size must be positive integers.')
        return x

    def predict(self, X, nsample=2000, target='median', random_state=None,
                batch_size=256, restricted=False):
        r"""Generate time draws or patient-specific summaries in fit-input units.

        :param X: Finite covariates with the same feature order used during fit.
        :type X: array-like of shape (n, input_dim)
        :param nsample: Positive Monte Carlo draws per patient; default 2000.
        :type nsample: int, optional
        :param target: 'mean' returns draw means; 'median' returns PyTorch sample medians; a numeric
            value in [0, 1] returns that quantile; None returns all time draws; 'log_samples'
            returns raw log-time draws.
        :type target: str, float, int or None, optional
        :param random_state: Temporary prediction seed. A numeric seed restores saved NumPy and
            PyTorch random states afterward. None uses the current random states.
        :type random_state: int or None, optional
        :param batch_size: Patients predicted per batch; default 256.
        :type batch_size: int, optional
        :param restricted: Cap time draws at the fitted horizon before time summaries. This does not
            affect training or raw log_samples.
        :type restricted: bool, optional

        :returns: Patient summaries, or the full draw matrix for target=None and target=log_samples.
        :rtype: numpy.ndarray of shape (n,) or (n, nsample)

        :raises ValueError: The target or checked inputs are invalid.
        :raises FloatingPointError: Exponentiating uncapped draws produces nonfinite time values.

        Time draws use :math:`\exp(U)\,\mathrm{time\_scale}`. Restriction replaces these
        with their minimum with :math:`\tau`. For an even number of draws, the median
        branch preserves ``torch.median`` behavior (the lower middle value).
        Values beyond the horizon are working-model extrapolations.
        """
        x = self._prediction_inputs(X, nsample, batch_size)
        valid = target in ('mean', 'median', 'log_samples', None) if isinstance(target, str) or target is None else False
        quantile = isinstance(target, (float, int, np.floating)) and not isinstance(target, bool) and 0 <= target <= 1
        if not (valid or quantile):
            raise ValueError("target must be 'mean', 'median', 'log_samples', a quantile in [0,1], or None.")
        if len(x) == 0:
            return np.empty((0, nsample)) if target in (None, 'log_samples') else np.empty(0)
        results = []
        with _prediction_rng(random_state):
            self.model.eval()
            with torch.no_grad():
                for start in range(0, len(x), batch_size):
                    xb = torch.as_tensor(x[start:start + batch_size], device=self.device)
                    u = self.model(xb, self._sample_noise(len(xb), nsample, self.noise_dim))
                    if target == 'log_samples':
                        result = u
                    else:
                        if restricted:
                            u = u.clamp_max(self.log_tau_)
                        samples = torch.exp(u.to(torch.float64)) * self.time_scale_
                        if not torch.isfinite(samples).all():
                            raise FloatingPointError("Exponentiated predictions overflow; inspect target='log_samples' or use restricted=True.")
                        if target is None:
                            result = samples
                        elif target == 'mean':
                            result = samples.mean(1)
                        elif target == 'median':
                            result = samples.median(1).values
                        else:
                            result = torch.quantile(samples, float(target), dim=1)
                    results.append(result.cpu().numpy())
        return np.concatenate(results, axis=0)

    def predict_survival(self, X, times, nsample=2000, random_state=None,
                         batch_size=256, restricted=False, allow_extrapolation=False):
        r"""Estimate survival probabilities by Monte Carlo comparisons on the log scale.

        :param X: Finite patient covariates in the fitted feature order.
        :type X: array-like of shape (n, input_dim)
        :param times: Finite nonnegative evaluation times in the same units as Y supplied to fit.
        :type times: array-like of shape (k,)
        :param nsample: Positive number of generated log-time draws per patient; default 2000.
        :type nsample: int, optional
        :param random_state: Temporary prediction seed; numeric seeds restore saved NumPy and
            PyTorch states on exit.
        :type random_state: int or None, optional
        :param batch_size: Number of patients evaluated per batch; default 256.
        :type batch_size: int, optional
        :param restricted: Return survival of the horizon-restricted predictive law, assigning zero
            at times greater than or equal to tau.
        :type restricted: bool, optional
        :param allow_extrapolation: Permit raw survival queries at or beyond tau. Required for such
            queries unless restricted=True; default False.
        :type allow_extrapolation: bool, optional

        :returns: Estimated survival probabilities in the order of the supplied query times.
        :rtype: numpy.ndarray of shape (n, k)

        :raises ValueError: Times are invalid, prediction inputs are invalid, or unsupported raw extrapolation is requested.

        Below the horizon, the Monte Carlo estimator is

        .. math::

           \widehat S_\nu(t\mid x)=\frac{1}{M}\sum_{j=1}^{M}
           \mathbf{1}\{U_\nu(x,\epsilon_j)>\log(t/\mathrm{time\_scale})\}.

        At zero the log threshold is negative infinity. Sorted draws and right-sided
        searches implement the strict survival inequality without a three-dimensional
        patient-by-draw-by-time allocation. Zero restricted survival at the horizon
        follows from the restriction definition and does not indicate observed failure.
        """
        x = self._prediction_inputs(X, nsample, batch_size)
        grid = np.asarray(times, dtype=np.float64)
        if grid.ndim != 1 or not np.isfinite(grid).all() or (grid < 0).any():
            raise ValueError('times must be a finite nonnegative one-dimensional array.')
        if not restricted and not allow_extrapolation and (grid >= self.tau_).any():
            raise ValueError('Raw survival at/above tau is extrapolation; set allow_extrapolation=True, or restricted=True for the restricted law.')
        if len(x) == 0:
            return np.empty((0, len(grid)))
        log_grid = np.full_like(grid, -np.inf)
        positive = grid > 0
        log_grid[positive] = np.log(grid[positive] / self.time_scale_)
        threshold = torch.as_tensor(log_grid, device=self.device)
        results = []
        with _prediction_rng(random_state):
            self.model.eval()
            with torch.no_grad():
                for start in range(0, len(x), batch_size):
                    xb = torch.as_tensor(x[start:start + batch_size], device=self.device)
                    u = self.model(xb, self._sample_noise(len(xb), nsample, self.noise_dim))
                    # Sorting and searchsorted avoid a batch*nsample*grid tensor.
                    ordered = u.to(torch.float64).sort(dim=1).values.contiguous()
                    thresholds = threshold.expand(len(xb), -1).contiguous()
                    count = torch.searchsorted(ordered, thresholds, right=True)
                    survival = 1.0 - count.to(torch.float64) / nsample
                    if restricted:
                        survival[:, torch.as_tensor(grid >= self.tau_, device=self.device)] = 0.
                    results.append(survival.cpu().numpy())
        return np.vstack(results)

EnSurvNet = _EnSurvNet
