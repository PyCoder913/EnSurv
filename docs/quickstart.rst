Quickstart
==========

Install EnSurv
--------------

EnSurv requires Python 3.11 or newer. After the first PyPI release, install the
model with:

.. code-block:: bash

   python -m pip install ensurv

For a source checkout, run ``python -m pip install .`` from the repository root.
Add ``[workflow]`` to the install target to enable preprocessing, metrics, and
diagnostic plotting. To use a specific CPU or accelerator build, install
`PyTorch <https://pytorch.org/get-started/locally/>`_ for your platform first.

Fit on synthetic data
---------------------

Give :meth:`ensurv.EnSurv.fit` a finite numeric feature matrix ``X``, strictly
positive observed follow-up times ``Y``, and binary event indicators ``Delta``.
Use ``1`` for an observed event and ``0`` for right censoring. Preserve the
training feature order when predicting. Fit any imputation, encoding, or
scaling on training data before applying it to held-out data.

.. code-block:: python

   import numpy as np
   from ensurv import EnSurv

   rng = np.random.default_rng(7)
   X = rng.normal(size=(128, 2))
   event_time = rng.lognormal(4.0 + 0.5 * X[:, 0], 0.4)
   censor_time = rng.uniform(25, 140, size=len(X))
   Y = np.minimum(event_time, censor_time)
   Delta = (event_time <= censor_time).astype(int)

   model = EnSurv(device="cpu", random_state=7)
   model.fit(X, Y, Delta, epochs=10, m=8, B=8)

Generate predictions
--------------------

``predict_survival`` returns one row per observation and one column per query
time. The default horizon ``model.tau_`` is the largest observed training
event time. ``endpoint=False`` keeps the query grid strictly below it.

.. code-block:: python

   times = np.linspace(0, model.tau_, 50, endpoint=False)
   survival = model.predict_survival(X[:3], times, nsample=500, random_state=7)
   medians = model.predict(X[:3], target="median", restricted=True, random_state=7)
   draws = model.predict(X[:3], target=None, nsample=500, restricted=True, random_state=7)

   print(survival.shape, medians.shape, draws.shape)
   # (3, 50) (3,) (3, 500)

Prediction times use the units supplied to ``fit``. ``restricted=True`` caps
time draws at the fitted horizon. The :doc:`method` guide explains how to
interpret predictions at that boundary.
