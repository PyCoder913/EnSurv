"""Population calibration and individual prediction visual diagnostics."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter


class EnSurvDiagnostics:
    r"""Produce the population and individual survival visual diagnostics.

    :param model: Fitted model trained on externally scaled times. It must provide predict and
        predict_survival methods.
    :type model: EnSurv or compatible predictor
    :param y_max: External time divisor used before fitting. Survival query times are divided by
        it, and time draws are multiplied by it to recover days; default 1.0.
    :type y_max: float, optional
    """
    def __init__(self, model, y_max=1.0):
        r"""Store the fitted model and external scale and apply the notebook seaborn theme.

        :param model: Fitted model trained on externally scaled times. It must provide predict and
            predict_survival methods.
        :type model: EnSurv or compatible predictor
        :param y_max: External time divisor used before fitting. Survival query times are divided by
            it, and time draws are multiplied by it to recover days; default 1.0.
        :type y_max: float, optional
        """
        self.model = model
        self.y_max = y_max
        sns.set_theme(style="ticks", context="paper", font_scale=1.2)

    def plot_population_calibration(self, X_test, Y_test, Delta_test, times, center_labels=None):
        r"""Compare mean predicted survival with global or region-specific observed KM curves.

        :param X_test: Test covariates in fitted feature order with the required training-fold
            scaling already applied.
        :type X_test: numpy.ndarray of shape (n_test, p)
        :param Y_test: Original unscaled test follow-up times in days.
        :type Y_test: numpy.ndarray of shape (n_test,)
        :param Delta_test: Original test failure indicators: 1 for failure and 0 for censoring.
        :type Delta_test: numpy.ndarray of shape (n_test,)
        :param times: Evaluation times in unscaled days, divided by self.y_max for prediction.
        :type times: numpy.ndarray of shape (k,)
        :param center_labels: Region/group labels for small-multiple calibration. None draws a
            single pooled comparison.
        :type center_labels: numpy.ndarray of shape (n_test,) or None, optional

        :returns: Global calibration figure or grid of group-specific calibration panels.
        :rtype: matplotlib.figure.Figure

        Survival calls retain allow_extrapolation=True and the model defaults for the number of
        draws and prediction seed; no new seed is introduced.
        """
        scaled_times = times / self.y_max
        
        if center_labels is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            kmf = KaplanMeierFitter()
            kmf.fit(Y_test, event_observed=Delta_test, label="Empirical KM (95% CI)")
            kmf.plot_survival_function(ax=ax, color='#e74c3c', linewidth=2)
            
            S_pred_matrix = self.model.predict_survival(X_test, scaled_times, allow_extrapolation=True)
            ax.plot(times, S_pred_matrix.mean(axis=0), label="EnSurv Predicted Average", color='#2980b9', linestyle='--', linewidth=2.5)
            
            ax.set_title("Population Survival Calibration", fontweight='bold')
            ax.set_xlabel("Time (Days)")
            ax.set_ylabel("Survival Probability S(t)")
            ax.legend(frameon=False)
            sns.despine()
            
        else:
            unique_centers = np.unique(center_labels)
            n_centers = len(unique_centers)
            
            # Dynamically calculate grid dimensions (e.g., 3x4 for 11 regions)
            cols = int(np.ceil(np.sqrt(n_centers)))
            rows = int(np.ceil(n_centers / cols))
            
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), sharex=True, sharey=True)
            axes_flat = axes.flatten() if n_centers > 1 else [axes]
            colors = sns.color_palette("husl", n_centers)
            
            for idx, center in enumerate(unique_centers):
                ax = axes_flat[idx]
                mask = (center_labels == center)
                
                # Plot Empirical
                kmf = KaplanMeierFitter()
                kmf.fit(Y_test[mask], event_observed=Delta_test[mask])
                kmf.plot_survival_function(ax=ax, color=colors[idx], ci_alpha=0.2, label=f"Empirical KM")
                
                # Plot Predicted
                S_pred_matrix = self.model.predict_survival(X_test[mask], scaled_times, allow_extrapolation=True)
                ax.plot(times, S_pred_matrix.mean(axis=0), color='black', linestyle='--', linewidth=2, label="Predicted Avg")
                
                ax.set_title(f"Region: {center}", fontsize=11, fontweight='bold')
                ax.legend(frameon=False, loc='lower left', fontsize=9)
                sns.despine(ax=ax)
                
            # Turn off empty axes if the grid isn't perfectly filled
            for idx in range(n_centers, len(axes_flat)):
                axes_flat[idx].axis('off')
                
            fig.supxlabel("Time (Days)", fontweight='bold')
            fig.supylabel("Survival Probability S(t)", fontweight='bold')
            plt.tight_layout()
            
        return fig

    def plot_individual_densities(self, X_test, patient_indices, true_times=None, true_events=None, nsample=5000, figsize=(10, 8)):
        # Draw raw samples and multiply by y_max to project them back into real time
        r"""Plot KDE summaries of generated time draws for selected patients.

        :param X_test: Test covariates in fitted feature order with the required training-fold
            scaling already applied.
        :type X_test: numpy.ndarray of shape (n_test, p)
        :param patient_indices: Nonempty positions of patients to plot within X_test; supplied
            outcome arrays are indexed by the same positions.
        :type patient_indices: sequence[int]
        :param true_times: Unscaled follow-up times in days for observed-time markers. Markers are
            drawn only when both true_times and true_events are supplied.
        :type true_times: numpy.ndarray of shape (n_test,) or None, optional
        :param true_events: Original failure indicators for choosing failure or censoring markers.
        :type true_events: numpy.ndarray of shape (n_test,) or None, optional
        :param nsample: Monte Carlo draws per plotted patient; default 5000. These density methods
            pass prediction random_state=42 and batch_size=256.
        :type nsample: int, optional
        :param figsize: Figure width and height in inches; default (10, 8).
        :type figsize: tuple[float, float], optional

        :returns: One density panel per patient, with generated median and optional observed-time
            markers.
        :rtype: matplotlib.figure.Figure

        Generated times are multiplied by self.y_max. KDE uses the original seaborn options,
        including cut=0. Censored follow-up is shown as an event-free interval, rather than as an
        observed failure time.
        """
        samples = self.model.predict(X_test[patient_indices], nsample=nsample, target=None, random_state=42, batch_size=256)
        samples = samples * self.y_max 
        
        fig, axes = plt.subplots(len(patient_indices), 1, figsize=figsize, sharex=True)
        if len(patient_indices) == 1: 
            axes = [axes]
        
        colors = sns.color_palette("crest", n_colors=len(patient_indices))
        
        for i, (idx, ax) in enumerate(zip(patient_indices, axes)):
            patient_samples = samples[i]
            
            # Added a label to the KDE plot for completeness in the legend
            sns.kdeplot(patient_samples, ax=ax, fill=True, color=colors[i], alpha=0.6, linewidth=1.5, label='Predicted Density', cut=0)
            
            median_val = np.median(patient_samples)
            ax.axvline(median_val, color=colors[i], linestyle=':', linewidth=2, label=f'Pred Median: {median_val:.1f}')
            
            if true_times is not None and true_events is not None:
                t_val = true_times[idx] # Assuming true_times is passed unscaled
                if true_events[idx] == 1:
                    ax.axvline(t_val, color='#c0392b', linestyle='-', linewidth=2.5, label=f'True Event: {t_val:.1f}')
                else:
                    ax.axvline(t_val, color='#7f8c8d', linestyle='-', linewidth=2.5, label=f'Censored: {t_val:.1f}')
                    # Added a label to the shaded region so it appears in the legend
                    ax.axvspan(0, t_val, color='#95a5a6', alpha=0.2, label='Event-Free Period') 
            
            ax.set_ylabel(f"Patient {idx}")
            ax.set_yticks([])
            
            # This is correctly indented inside the for-loop to apply to every subplot
            ax.legend(loc='upper right', frameon=False, fontsize=9)
            sns.despine(left=True)
            
        axes[-1].set_xlabel("Generated Survival Time Distribution (Days)")
        fig.suptitle("Individual Posterior Survival Densities (EnSurv)", fontweight='bold', y=0.98)
        plt.tight_layout()
        
        return fig
        
    def plot_prediction_scatter(self, X_test, Y_test, Delta_test, figsize=(8, 8), max_points=200, random_seed=42):
        r"""Plot predicted median times against observed failure or censoring times.

        :param X_test: Test covariates in fitted feature order with the required training-fold
            scaling already applied.
        :type X_test: numpy.ndarray of shape (n_test, p)
        :param Y_test: Original unscaled test follow-up times in days.
        :type Y_test: numpy.ndarray of shape (n_test,)
        :param Delta_test: Original test failure indicators: 1 for failure and 0 for censoring.
        :type Delta_test: numpy.ndarray of shape (n_test,)
        :param figsize: Figure width and height in inches; default (8, 8).
        :type figsize: tuple[float, float], optional
        :param max_points: Maximum plotted patients. Larger cohorts are sampled without replacement
            before prediction; None uses all rows. Default 200.
        :type max_points: int or None, optional
        :param random_seed: NumPy seed set globally when subsampling is needed; default 42. This
            method does not pass a separate prediction seed.
        :type random_seed: int, optional

        :returns: Scatter figure with distinct failure/censoring markers and an identity reference
            line.
        :rtype: matplotlib.figure.Figure

        Predicted medians are converted to days using self.y_max. Observed times for censored
        patients are follow-up bounds rather than true failure times.
        """
        # Subsample to avoid clutter before running predictions
        if max_points is not None and len(Y_test) > max_points:
            np.random.seed(random_seed)
            indices = np.random.choice(len(Y_test), max_points, replace=False)
            X_plot = X_test[indices]
            Y_plot = Y_test[indices]
            Delta_plot = Delta_test[indices]
        else:
            X_plot = X_test
            Y_plot = Y_test
            Delta_plot = Delta_test

        preds = self.model.predict(X_plot, target="median") * self.y_max
        
        fig, ax = plt.subplots(figsize=figsize)
        
        events = Delta_plot == 1
        ax.scatter(Y_plot[events], preds[events], alpha=0.8, color='#2c3e50', edgecolor='w', s=60, label='Observed Graft Failure')
        
        censored = Delta_plot == 0
        ax.scatter(Y_plot[censored], preds[censored], alpha=0.6, color='#bdc3c7', marker='^', s=60, label='Censored (Time to End of Study)')
        
        max_val = max(Y_plot.max(), preds.max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='x=y (Perfect Exact Prediction)')
        
        ax.fill_between([0, max_val], [0, max_val], max_val, color='#2ecc71', alpha=0.05, label='Valid Region for Censored')
        
        ax.set_title(f"Predicted vs Observed Survival Times (Subsample N={len(Y_plot)})", fontweight='bold')
        ax.set_xlabel("Observed Time (Days)")
        ax.set_ylabel("Predicted Median Time (Days)")
        ax.legend(frameon=True, loc='upper left')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        sns.despine()
        return fig

    def plot_individual_densities_grid(self, X_test, patient_indices, 
                                   true_times=None, true_events=None, 
                                   nsample=5000, num_grid_points=250, figsize=(10, 8)):
        # 1. Define a time grid starting strictly at 0
        r"""Estimate plotted time densities by finite differences of generated survival curves.

        :param X_test: Test covariates in fitted feature order with the required training-fold
            scaling already applied.
        :type X_test: numpy.ndarray of shape (n_test, p)
        :param patient_indices: Nonempty positions of patients to plot within X_test; supplied
            outcome arrays are indexed by the same positions.
        :type patient_indices: sequence[int]
        :param true_times: Unscaled follow-up times in days for observed-time markers. Markers are
            drawn only when both true_times and true_events are supplied.
        :type true_times: numpy.ndarray of shape (n_test,) or None, optional
        :param true_events: Original failure indicators for choosing failure or censoring markers.
        :type true_events: numpy.ndarray of shape (n_test,) or None, optional
        :param nsample: Monte Carlo draws per plotted patient; default 5000. These density methods
            pass prediction random_state=42 and batch_size=256.
        :type nsample: int, optional
        :param num_grid_points: Points in the evenly spaced time grid from zero to 1.1 times the
            maximum true_times, or 1.1 times self.y_max when true_times is None; default 250.
        :type num_grid_points: int, optional
        :param figsize: Figure width and height in inches; default (10, 8).
        :type figsize: tuple[float, float], optional

        :returns: Patient density panels on time-grid midpoints with grid-based median and optional
            observed-time markers.
        :rtype: matplotlib.figure.Figure

        .. math::

           \widehat f(t_{j+1/2})=
           \frac{\widehat S(t_j)-\widehat S(t_{j+1})}{t_{j+1}-t_j}.

        Prediction retains allow_extrapolation=True. The plotted median is the grid
        point whose survival probability is nearest to one half.
        """
        max_t = true_times.max() * 1.1 if true_times is not None else self.y_max * 1.1
        t_grid = np.linspace(0, max_t, num_grid_points)
        
        # Scale grid down to the model's latent scale (assuming model outputs scaled logT)
        t_grid_scaled = t_grid / self.y_max 
        
        # 2. Predict survival probabilities S(t) directly on the grid
        # Shape: (len(patient_indices), num_grid_points)
        S_pred = self.model.predict_survival(
            X_test[patient_indices], 
            times=t_grid_scaled, 
            nsample=nsample, 
            random_state=42, 
            batch_size=256, allow_extrapolation=True
        )
        
        # 3. Calculate density via negative finite differences: f(t) ≈ -(S(t + dt) - S(t)) / dt
        dt = np.diff(t_grid) 
        pdf = (S_pred[:, :-1] - S_pred[:, 1:]) / dt
        
        # Use grid midpoints for plotting the density
        t_mid = (t_grid[:-1] + t_grid[1:]) / 2
        
        fig, axes = plt.subplots(len(patient_indices), 1, figsize=figsize, sharex=True)
        if len(patient_indices) == 1: 
            axes = [axes]
            
        colors = sns.color_palette("crest", n_colors=len(patient_indices))
        
        for i, (idx, ax) in enumerate(zip(patient_indices, axes)):
            patient_pdf = pdf[i]
            
            # Plot the grid-based density
            ax.plot(t_mid, patient_pdf, color=colors[i], linewidth=1.5, label='Grid-Based Density')
            ax.fill_between(t_mid, 0, patient_pdf, color=colors[i], alpha=0.6)
            
            # Approximate median directly from the S(t) grid
            idx_median = np.argmin(np.abs(S_pred[i] - 0.5))
            median_val = t_grid[idx_median]
            ax.axvline(median_val, color=colors[i], linestyle=':', linewidth=2, label=f'Pred Median: {median_val:.1f}')
            
            if true_times is not None and true_events is not None:
                t_val = true_times[idx]
                if true_events[idx] == 1:
                    ax.axvline(t_val, color='#c0392b', linestyle='-', linewidth=2.5, label=f'True Event: {t_val:.1f}')
                else:
                    ax.axvline(t_val, color='#7f8c8d', linestyle='-', linewidth=2.5, label=f'Censored: {t_val:.1f}')
                    ax.axvspan(0, t_val, color='#95a5a6', alpha=0.2, label='Event-Free Period')
            
            ax.set_ylabel(f"Patient {idx}")
            ax.set_yticks([])
            ax.legend(loc='upper right', frameon=False, fontsize=9)
            sns.despine(left=True)
            
        plt.tight_layout()
        return fig

