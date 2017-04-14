"""
This module provides Exogenous variable modeling for use in other forecasting models.
Basically it fits a set of univariate Autoregressive models to each exogenous series
and produces a dataframe with forecasted exogenous variables on demand
"""

import numpy as np
import dask
from dask import delayed
import pandas as pd
import pyflux as pf
import matplotlib.pyplot as plt
from infodenguepredict.data.infodengue import get_alerta_table, get_temperature_data, get_tweet_data, build_multicity_dataset


class ExogenousForecast:
    def __init__(self, exog: pd.DataFrame, dists=None):
        """
        Exogenous variables forecaster
        :param exog: Dataframe containing exogenous variables
        :param dists: distributions of each series e.g. ['poisson', 'gaussian', ...]
        """
        self.exog = exog
        self.models = {n: None for n in exog.columns}
        self.fits = {n: None for n in exog.columns}
        self.fitted = False
        if dists is None:
            self.dists = ['gaussian' for i in exog.columns]
        else:
            try:
                assert len(dists) == len(exog.columns)
                self.dists = dists
            except AssertionError:
                raise AssertionError("You must provide a distribution for each series")

    def _fit(self):
        print("Starting to fit {} models".format(len(self.dists)))
        for series, dist in zip(self.exog.columns, self.dists):
            if dist == 'gaussian':
                self.models[series] = build_GAS_model(data=self.exog, target=series, family=pf.families.Normal)
            else:
                self.models[series] = build_GAS_model(data=self.exog, target=series)
        for n, m in self.models.items():
            self.fits[n] = delayed(m.fit)()
        dask.compute(*self.models.values())
        self.fitted = True

    def print_fit_summary(self, vname):
        """
        Prints the summary of the fit of a given exogenous variable
        :param vname: exogenous variable name present in self.exog
        :return: None
        """
        if not self.fitted:
            print("You must fit the models first. Run 'get_forecast' first")
            return
        print(self.fits[vname].summary())

    def get_forecast(self, N: int) -> pd.DataFrame:
        """
        Returns a Dataframe with the N-steps forecasts for all the exogenous variables.
        :param N: 
        :return: pandas Dataframe with forecasted series.
        """
        print("Generating forecasts")
        if not self.fitted:
            self._fit()
        forecasts = {}
        for n, m in self.models.items():
            forecasts[n] = m.predict(N) # delayed(m.predict)(N)
        # dask.compute(*forecasts.values())
        # print(m.predict(N))
        return pd.concat(forecasts.values(), axis=1)



def build_ARIMA_model():
    def build_model(data, ar=4, ma=4, integ=0, target=None):
        model = pf.ARIMA(data=data, ar=2, ma=1, integ=1, target=target)
        return model

def build_GAS_model(data, ar=2, sc=4, family=pf.families.Poisson, target=None):
    model = pf.GAS(data=data, ar=ar, sc=sc, family=family(), target=target)
    return model


if __name__ == "__main__":
    data = build_multicity_dataset('RJ')
    EF = ExogenousForecast(data[data.columns[:3]], ['poisson']*3)
    df = EF.get_forecast(6)
    print(df)
