# EnSurv

EnSurv fits a generative model for right-censored time-to-event data. A PyTorch generator learns to draw event times conditional on covariates. These draws can be summarized as patient-level survival curves, medians, quantiles, or sample distributions.

## Installation

After the first PyPI release, install EnSurv with Python 3.11 or newer.

```bash
python -m pip install ensurv
```

From a source checkout, use `python -m pip install .`. Add the `workflow` extra for data preprocessing, evaluation metrics, and diagnostic plots (`ensurv[workflow]` or `.[workflow]`).

## Fit and predict

```python
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

times = np.linspace(0, model.tau_, 50, endpoint=False)
survival = model.predict_survival(X[:3], times, nsample=500, random_state=7)
print(survival.shape)  # (3, 50)
```

`Y` contains positive observed follow-up times. `Delta=1` marks an observed event and `Delta=0` marks right censoring. The default fitted horizon `model.tau_` is the largest observed event time in the training data. Keep raw survival queries below that horizon or pass `restricted=True` for the horizon-restricted distribution.

## Examples and documentation

- `examples/EnSurv_SUPPORT_Example.ipynb` walks through three-fold cross-validation, metrics, and diagnostics on public SUPPORT data.
- `simulations/` contains the asymptotic and synthetic prediction experiments.
- `docs/` contains the Sphinx user guide and API reference. Build it with `python -m pip install ".[docs,workflow]"` followed by `python -m sphinx -W -b html docs docs/_build/html`.
- `RELEASE.md` covers building distributions, publishing to PyPI, and hosting the docs on Read the Docs.
