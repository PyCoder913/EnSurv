Examples and simulations
========================

SUPPORT cross-validation
------------------------

:download:`Download the SUPPORT notebook <../examples/EnSurv_SUPPORT_Example.ipynb>`
for a three-fold workflow using public cohort data. It estimates time to death
in days from recorded covariates and shows metrics, training diagnostics,
and fitted feature summaries. The notebook includes data provenance and the
feature list.

After publication, install the workflow and notebook dependencies with:

.. code-block:: bash

   python -m pip install "ensurv[workflow]" jupyterlab
   python -m jupyterlab

From a checkout, replace ``ensurv[workflow]`` with ``.[workflow]`` in that
command. Open the notebook in JupyterLab and select the kernel from the same
environment. It downloads the public data on the first run and saves fold
artifacts under ``support_cv_outputs/``. For a smaller run, set
``N_PATIENTS = 600`` and ``EPOCHS = 5`` in the configuration cell.

Each fold fits preprocessing statistics on its training rows and evaluates
on held-out rows. Keep each fold's model, time divisor, and feature order
together when interpreting predictions and diagnostic plots.

Simulation notebooks
--------------------

* :download:`Asymptotic simulations <../simulations/Asymptotic_Simulations.ipynb>`
  exercise the Kaplan–Meier completion and energy-score routines. Install
  ``ensurv`` with ``matplotlib``, ``scipy``, and ``jupyterlab``.
* :download:`Prediction simulations <../simulations/Prediction_Simulations.ipynb>`
  fit EnSurv across synthetic data-generating scenarios and export a results
  CSV. Install ``ensurv[workflow]`` with ``jupyterlab``.
