Model and horizon
=================

EnSurv models a conditional distribution of event times from features and
random noise. Its generator operates on the log-time scale. Log-time draws
can be converted to event times, summarized with quantiles, or compared with
a time grid to estimate a survival curve.

Horizon restriction and completion
----------------------------------

A fit uses a horizon :math:`\tau`. Set it explicitly in the same units as
``Y`` or use the default ``tau="max_event"``, which selects the largest
observed event time in the training data. Each outcome contributes to the
energy score according to its follow-up:

* An observed event before :math:`\tau` contributes its observed time.
* Follow-up reaching :math:`\tau` contributes the restricted time :math:`\tau`.
* A right-censored observation before :math:`\tau` is scored against the
  part of a pooled Kaplan–Meier completion distribution strictly after its
  censoring time. Bootstrap estimates average the contribution, with residual
  probability mass assigned to :math:`\tau`.

The fitted generator targets the resulting completed working distribution.
Its agreement with the underlying conditional event-time distribution depends
on how well pooled completion represents the censored outcomes.

.. seealso::

   :func:`ensurv.km.kaplan_meier_batched` documents the bootstrap curves.
   :func:`ensurv.loss.survival_engression_loss` gives the empirical score.

Prediction at the horizon
-------------------------

By default, ``predict_survival`` accepts only times strictly below the fitted
horizon. With ``restricted=True``, it estimates the survival function of
:math:`\min(T,\tau)` and returns zero at and after :math:`\tau` by definition.
To query raw generator survival at or beyond the horizon, pass
``allow_extrapolation=True`` and interpret those estimates as extrapolations of
the fitted working model. Raw time draws can also extend past the horizon.

Time scaling and evaluation
---------------------------

The model's ``time_scale`` option divides durations internally before taking
logs. The fit and prediction APIs continue to accept times in the original
fit-input units. The :doc:`SUPPORT workflow <examples>` uses a different
convention for :func:`ensurv.evaluation.compute_all_metrics`: it divides the
training times externally by the training maximum ``y_max`` and provides
original, day-scale outcomes plus ``y_max`` to the evaluator. In that
workflow, leave the model's internal ``time_scale`` at its default of 1.

The evaluator uses fixed one- and five-year horizons in days. It enables raw
extrapolation over its evaluation grid, so check the fold's fitted horizon
when interpreting its metrics.
