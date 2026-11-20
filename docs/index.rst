:html_theme.sidebar_secondary.remove: true

EnSurv
======

.. toctree::
   :hidden:
   :maxdepth: 2

   quickstart
   method
   examples
   api

.. container:: ensurv-hero

   .. container:: ensurv-intro

      .. container:: ensurv-kicker

         Generative survival analysis

      .. container:: ensurv-headline

         Learn the distribution of event times

      Fit EnSurv model to right-censored data. Draw possible event times, estimate
      patient-specific survival curves, and inspect the fitted model.

      .. button-ref:: quickstart
         :class: ensurv-cta

         Start with a CPU example →

   .. figure:: _static/survival-curves.png
      :class: dark-light
      :alt: Two illustrative patient survival curves fall at different rates. A vertical line marks the fitted horizon tau.

      Illustrative survival curves up to the fitted horizon.

.. code-block:: bash
   :class: ensurv-install

   python -m pip install ensurv

.. grid:: 1 2 2 2
   :gutter: 3
   :class-container: ensurv-guides

   .. grid-item-card:: Fit your first model
      :link: quickstart
      :link-type: doc

      Run a synthetic example and get survival probabilities, medians, and time draws.

   .. grid-item-card:: Understand the horizon
      :link: method
      :link-type: doc

      See how pooled Kaplan–Meier completion and horizon restriction enter the model.

   .. grid-item-card:: Explore worked notebooks
      :link: examples
      :link-type: doc

      Run the public SUPPORT workflow or reproduce the synthetic experiments.

   .. grid-item-card:: Browse the API
      :link: api
      :link-type: doc

      Look up model parameters, prediction shapes, metrics, and plotting helpers.
