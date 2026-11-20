import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from lifelines import KaplanMeierFitter


# Training diagnostics plotter

 
_PALETTE = ["#1b3a5c", "#c0442c", "#3f8f6f", "#8a6bb1", "#c99a2e", "#4a7fa8",
            "#9c5a2e", "#5a5a5a", "#2e7d7d", "#a13d63"]
 
 
def set_journal_style():
    r"""Apply the Matplotlib style defaults globally.

    Updates ``matplotlib.pyplot.rcParams`` for fonts, spines, grids, legends, figure
    backgrounds, and saving. It returns None and does not restore previous settings.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#1a1a1a",
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
 
 
# 1. Loss decomposition over training
def plot_loss_decomposition(history, save_path=None):
    r"""Plot total training loss and its recorded score components.

    :param history: Training diagnostics from EnSurv.fit with track_diagnostics=True, containing
        the keys used by this plot.
    :type history: dict
    :param save_path: Destination for saving the figure. None returns the figure without saving;
        parent directories must already exist.
    :type save_path: str or path-like or None, optional

    :returns: Figure and its two vertically stacked axes.
    :rtype: tuple[matplotlib.figure.Figure, tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]]

    The historical event_loss key also includes restricted outcomes known from follow-up to the
    horizon.
    """
    set_journal_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 6.0), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1.2]})
 
    ep = history["epoch"]
    ax1.plot(ep, history["total_loss"], color=_PALETTE[0], lw=1.8)
    ax1.set_ylabel("Total training loss")
    ax1.set_title("Training loss")
 
    ax2.plot(ep, history["event_loss"], color=_PALETTE[1], lw=1.5, label="Event term")
    ax2.plot(ep, history["censor_loss"], color=_PALETTE[2], lw=1.5, label="Censoring term (IPCW-style)")
    ax2.plot(ep, history["energy_term"], color=_PALETTE[3], lw=1.5, label="Energy (diversity) term")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Component value")
    ax2.set_title("Loss components")
    ax2.legend(loc="best")
 
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig, (ax1, ax2)
 
 
# 2. Gradient flow heatmap (epoch x layer group), log color scale
def plot_gradient_flow_heatmap(history, save_path=None, group_order=None):
    r"""Plot recorded parameter-group gradient RMS magnitudes on a log color scale.

    :param history: Training diagnostics from EnSurv.fit with track_diagnostics=True, containing
        the keys used by this plot.
    :type history: dict
    :param save_path: Destination for saving the figure. None returns the figure without saving;
        parent directories must already exist.
    :type save_path: str or path-like or None, optional
    :param group_order: Gradient-group labels in the requested row order. None follows insertion
        order in history[grad_norms].
    :type group_order: sequence[str] or None, optional

    :returns: Figure and heatmap axis.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    set_journal_style()
    grad = history["grad_norms"]
    groups = group_order if group_order is not None else list(grad.keys())
    mat = np.array([grad[g] for g in groups])  # (n_groups, n_epochs)
    mat = np.clip(mat, 1e-12, None)
 
    fig, ax = plt.subplots(figsize=(7.2, max(2.5, 0.42 * len(groups))))
    im = ax.imshow(mat, aspect="auto", cmap="magma",
                    norm=LogNorm(vmin=np.percentile(mat, 1), vmax=mat.max()),
                    interpolation="nearest",
                    extent=[history["epoch"][0], history["epoch"][-1], len(groups), 0])
    ax.set_yticks(np.arange(len(groups)) + 0.5)
    ax.set_yticklabels(groups)
    ax.set_xlabel("Epoch")
    ax.set_title("Gradient flow by layer (log scale)")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean gradient norm")
 
    if save_path:
        fig.savefig(save_path)
    return fig, ax
 
 
# 3. Feature importance evolution: 100%-stacked area (top-k + "other")
def plot_feature_importance_evolution(history, feature_names=None, top_k=8, save_path=None):
    r"""Plot stacked structural feature-importance percentages over training.

    :param history: Training diagnostics from EnSurv.fit with track_diagnostics=True, containing
        the keys used by this plot.
    :type history: dict
    :param feature_names: Names in the fitted feature order. None generates X0, X1, and so on.
    :type feature_names: sequence[str] or None, optional
    :param top_k: Number of features selected by their final recorded importance; default 8.
        Remaining features are pooled into an Other band.
    :type top_k: int, optional
    :param save_path: Destination for saving the figure. None returns the figure without saving;
        parent directories must already exist.
    :type save_path: str or path-like or None, optional

    :returns: Figure and stacked-area axis.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    set_journal_style()
    imp = np.asarray(history["importance_pct"])  # (n_snapshots, n_features)
    ep = history["importance_epochs"]
    n_features = imp.shape[1]
 
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(n_features)]
 
    final_rank = np.argsort(-imp[-1])
    top_idx = final_rank[:top_k]
    other_idx = final_rank[top_k:]
 
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    series = [imp[:, i] for i in top_idx]
    labels = [feature_names[i] for i in top_idx]
    if len(other_idx) > 0:
        series.append(imp[:, other_idx].sum(axis=1))
        labels.append(f"Other ({len(other_idx)} features)")
 
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(series) - (1 if len(other_idx) > 0 else 0))]
    if len(other_idx) > 0:
        colors.append("#bbbbbb")
 
    ax.stackplot(ep, series, labels=labels, colors=colors, alpha=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Feature importance (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Feature importance over training")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
 
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig, ax
 
 
# 4. Linear vs. nonlinear weight-norm evolution
def plot_weight_norm_evolution(history, save_path=None):
    r"""Compare recorded linear and nonlinear structural weight norms over training.

    :param history: Training diagnostics from EnSurv.fit with track_diagnostics=True, containing
        the keys used by this plot.
    :type history: dict
    :param save_path: Destination for saving the figure. None returns the figure without saving;
        parent directories must already exist.
    :type save_path: str or path-like or None, optional

    :returns: Figure and weight-norm axis.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    Plots :math:`\|\beta\|_1` when available and the summed nonlinear feature norms at
    importance snapshot epochs.
    """
    set_journal_style()
    wn = history["weight_norms"]
    ep = history.get("importance_epochs", history["epoch"])
 
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    if "beta_l1" in wn and wn["beta_l1"] is not None:
        ax.plot(ep, wn["beta_l1"], color=_PALETTE[0], lw=1.8, label=r"$\|\beta\|_1$ (linear term)")
    ax.plot(ep, wn["nonlinear_total"], color=_PALETTE[1], lw=1.8, label="Nonlinear path norm")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight norm")
    ax.set_title("Linear vs. nonlinear capacity allocation")
    ax.legend(loc="best")
 
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig, ax
 
# 5. Combined dashboard
def plot_training_dashboard(history, feature_names=None, top_k=6, save_path=None, suptitle="Post-Transplant Setting: EnSurv Training Diagnostics"):
    r"""Create the notebook four-panel training diagnostics dashboard.

    :param history: Training diagnostics from EnSurv.fit with track_diagnostics=True, containing
        the keys used by this plot.
    :type history: dict
    :param feature_names: Names in the fitted feature order. None generates X0, X1, and so on.
    :type feature_names: sequence[str] or None, optional
    :param top_k: Features shown separately in the stacked importance panel, ranked at the final
        snapshot; default 6.
    :type top_k: int, optional
    :param save_path: Destination for saving the figure. None returns the figure without saving;
        parent directories must already exist.
    :type save_path: str or path-like or None, optional
    :param suptitle: Figure-wide title, defaulting to the original post-transplant diagnostics
        title.
    :type suptitle: str, optional

    :returns: Figure and primary axes for loss, gradient flow, importance, and weight norms. The
        loss panel also creates a secondary axis.
    :rtype: tuple[matplotlib.figure.Figure, tuple[matplotlib.axes.Axes, ...]]
    """
    set_journal_style()
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2)
 
    ax1 = fig.add_subplot(gs[0, 0])
    ep = history["epoch"]
    ax1.plot(ep, history["total_loss"], color=_PALETTE[0], lw=1.8, label="Total")
    ax1b = ax1.twinx()
    ax1b.plot(ep, history["event_loss"], color=_PALETTE[1], lw=1.2, ls="--", label="Event")
    ax1b.plot(ep, history["censor_loss"], color=_PALETTE[2], lw=1.2, ls="--", label="Censor")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Total loss")
    ax1b.set_ylabel("Component value")
    ax1.set_title("(A) Training loss")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="best", fontsize=7.5)
 
    ax2 = fig.add_subplot(gs[0, 1])
    grad = history["grad_norms"]
    groups = list(grad.keys())
    mat = np.clip(np.array([grad[g] for g in groups]), 1e-12, None)
    im = ax2.imshow(mat, aspect="auto", cmap="magma",
                     norm=LogNorm(vmin=np.percentile(mat, 1), vmax=mat.max()),
                     extent=[ep[0], ep[-1], len(groups), 0])
    ax2.set_yticks(np.arange(len(groups)) + 0.5)
    ax2.set_yticklabels(groups, fontsize=8)
    ax2.set_xlabel("Epoch")
    ax2.set_title("(B) Gradient flow by layer")
    fig.colorbar(im, ax=ax2, pad=0.02, fraction=0.046)
 
    ax3 = fig.add_subplot(gs[1, 0])
    imp = np.asarray(history["importance_pct"])
    imp_ep = history["importance_epochs"]
    n_features = imp.shape[1]
    fn = feature_names if feature_names is not None else [f"X{i}" for i in range(n_features)]
    final_rank = np.argsort(-imp[-1])
    top_idx = final_rank[:top_k]
    other_idx = final_rank[top_k:]
    series = [imp[:, i] for i in top_idx]
    labels = [fn[i] for i in top_idx]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(top_idx))]
    if len(other_idx) > 0:
        series.append(imp[:, other_idx].sum(axis=1))
        labels.append("Other")
        colors.append("#bbbbbb")
    ax3.stackplot(imp_ep, series, labels=labels, colors=colors, alpha=0.9)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Importance (%)")
    ax3.set_ylim(0, 100)
    ax3.set_title("(C) Feature importance evolution")
    ax3.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)
 
    ax4 = fig.add_subplot(gs[1, 1])
    wn = history["weight_norms"]
    if "beta_l1" in wn and wn["beta_l1"] is not None:
        ax4.plot(imp_ep, wn["beta_l1"], color=_PALETTE[0], lw=1.8, label=r"$\|\beta\|_1$")
    ax4.plot(imp_ep, wn["nonlinear_total"], color=_PALETTE[1], lw=1.8, label="Nonlinear norm")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Weight norm")
    ax4.set_title("(D) Linear vs. nonlinear capacity")
    ax4.legend(loc="best", fontsize=8)
 
    fig.suptitle(suptitle, fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path)
    return fig, (ax1, ax2, ax3, ax4)


def plot_time_calibration(S_pred, times_grid, y_te, e_te, target_time, n_bins=10, ax=None):
    r"""Compare quantile-binned predictions with observed KM survival at one time.

    :param S_pred: Patient-by-time survival predictions.
    :type S_pred: numpy.ndarray of shape (n_test, k)
    :param times_grid: Evaluation times in days, corresponding to columns of S_pred.
    :type times_grid: numpy.ndarray of shape (k,)
    :param y_te: Original test follow-up times in days.
    :type y_te: numpy.ndarray of shape (n_test,)
    :param e_te: Original test failure indicators.
    :type e_te: numpy.ndarray of shape (n_test,)
    :param target_time: Requested calibration time in days. The nearest evaluation-grid column
        supplies predictions, while observed KM is evaluated at this exact value.
    :type target_time: float
    :param n_bins: Requested number of prediction quantile bins; default 10. Duplicate bin edges
        are removed.
    :type n_bins: int, optional
    :param ax: Axis to draw on. None creates a new figure and axis.
    :type ax: matplotlib.axes.Axes or None, optional

    :returns: Axis with the calibration curve and identity reference line.
    :rtype: matplotlib.axes.Axes

    Both edges of each bin are included, preserving the original handling of boundary ties.
    Labels express the target in years using 365.25 days per year.
    """
    # 1. Locate the exact index for the target time in our predictions
    # (Since we injected target_time into times_grid, it will match exactly)
    time_idx = np.argmin(np.abs(times_grid - target_time))
    
    # 2. Extract predicted survival probabilities for all patients at this time
    preds_at_t = S_pred[:, time_idx]
    
    # 3. Bin predictions into quantiles (e.g., deciles)
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(preds_at_t, quantiles)
    # Handle duplicate edges if many predictions are identical
    bin_edges = np.unique(bin_edges) 
    
    # 4. Calculate observed vs predicted in each bin
    kmf = KaplanMeierFitter()
    observed_surv = []
    expected_surv = []
    
    for i in range(len(bin_edges) - 1):
        # Find patients in the current probability bin
        mask = (preds_at_t >= bin_edges[i]) & (preds_at_t <= bin_edges[i+1])
        if not mask.any():
            continue
            
        expected_surv.append(preds_at_t[mask].mean())
        
        # Fit KM on this specific subset of patients
        kmf.fit(y_te[mask], event_observed=e_te[mask])
        
        # Get KM survival estimate at the target time
        # If target_time > max time in this bin, forward-fill the last known survival
        obs = kmf.predict(target_time)
        observed_surv.append(obs)
        
    # 5. Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    ax.plot(expected_surv, observed_surv, marker='o', linestyle='-', label=f'Model at {target_time/365.25:.0f} Yr')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    
    ax.set_xlabel(f"Predicted Survival Probability at {target_time/365.25:.0f} Years")
    ax.set_ylabel(f"Observed KM Survival at {target_time/365.25:.0f} Years")
    ax.set_title(f"Calibration Plot at {target_time/365.25:.0f} Years")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_pooled_calibration(preds, y_true, e_true, target_time, title, n_bins=100, ax=None):
    r"""Plot one-time calibration after pooling held-out predictions across folds.

    :param preds: Survival probabilities at the requested time for pooled held-out patients.
    :type preds: numpy.ndarray of shape (n,)
    :param y_true: Original follow-up times in days.
    :type y_true: numpy.ndarray of shape (n,)
    :param e_true: Original binary failure indicators.
    :type e_true: numpy.ndarray of shape (n,)
    :param target_time: Time in days at which within-bin observed KM survival is evaluated.
    :type target_time: float
    :param title: Title displayed on the calibration axis.
    :type title: str
    :param n_bins: Requested quantile bins; default 100. Duplicate edges are removed and both
        bin endpoints remain inclusive.
    :type n_bins: int, optional
    :param ax: Destination axis. None creates an axis unless preds is empty.
    :type ax: matplotlib.axes.Axes or None, optional

    :returns: Calibration axis; an empty prediction array returns the original ax unchanged.
    :rtype: matplotlib.axes.Axes or None
    """
    if len(preds) == 0:
        return ax
        
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(np.quantile(preds, quantiles))
    
    kmf = KaplanMeierFitter()
    observed_surv = []
    expected_surv = []
    
    for i in range(len(bin_edges) - 1):
        mask = (preds >= bin_edges[i]) & (preds <= bin_edges[i+1])
        if not mask.any():
            continue
            
        expected_surv.append(preds[mask].mean())
        
        kmf.fit(y_true[mask], event_observed=e_true[mask])
        observed_surv.append(kmf.predict(target_time))
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    ax.plot(expected_surv, observed_surv, linestyle='-', label='EnSurv (Pooled on 3-Fold CV)')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    
    ax.set_xlabel(f"Predicted Survival Probability", fontsize=14)
    ax.set_ylabel(f"Observed KM Survival", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    return ax

