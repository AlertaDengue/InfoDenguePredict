======
Models
======

This section documents the available models.

LSTM
----

This model is based on the Long Short Term Memory deep learning model, and is implemented using Keras.

Arima
-----

Simple ARIMA model implemented using the PyFlux library.

GAS
---

Simple GAS model implemented using the PyFlux library.

Vector auto regression
----------------------

The VAR model is implemented following the tutorial on PyFlux.

R_forecast
----------

This model uses the R library `forecast <https://cran.r-project.org/web/packages/forecast/index.html>`. So in order to
run it you must have R with the forecast package installed.

Currently the model adjusted to the curves is a simple arima, but more sophisticated models are available in R which can be tried.

Also this model can serve as a template for the implementation of other R based predictive models, using rpy2 as a
bridge between Python and R.
