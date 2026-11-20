API reference
=============

Fit and predict
---------------

.. autoclass:: ensurv.EnSurv
   :members: fit, predict, predict_survival, explain

.. autoclass:: ensurv.EnSurvNet
   :members: forward

Preprocessing and evaluation
----------------------------

Install the ``workflow`` extra to use these functions.

.. autofunction:: ensurv.preprocessing.preprocess_df
.. autofunction:: ensurv.preprocessing.prepare_unos_dataset
.. autofunction:: ensurv.evaluation.compute_all_metrics
.. autofunction:: ensurv.evaluation.analyze_hierarchical_importance

Diagnostics
-----------

.. autoclass:: ensurv.diagnostics.EnSurvDiagnostics
   :members:

.. automodule:: ensurv.utils
   :members:

Horizon and training score
--------------------------

.. automodule:: ensurv.horizon
   :members: resolve_tau, apply_tau_restriction

.. automodule:: ensurv.km
   :members:

.. automodule:: ensurv.loss
   :members:

Research workflows
------------------

For the original kidney experiment, supply your own dataset path explicitly.

.. autofunction:: ensurv.cross_validation.run_cross_validation

.. automodule:: ensurv.workflows
   :members:
