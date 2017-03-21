"""
Tests related to data fetching code
"""

import unittest
from infodenguepredict.data.infodengue import get_temperature_data, get_alerta_table, get_tweet_data, build_multicity_dataset
from infodenguepredict.data.satellite import  LandSurfaceTemperature
import pandas as pd
import os
from glob import glob


__author__ = 'fccoelho@gmail.com'


class TestInfodengue(unittest.TestCase):
    def test_temp_data(self):
        df = get_temperature_data(3303609)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(df.size, 0)

    def test_tweet(self):
        df = get_tweet_data(3303609)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(df.size, 0)

    def test_alerta_one_munic(self):
        df = get_alerta_table(3303609)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(df.size, 0)
        self.assertGreater(df.casos.sum(), 0)
        self.assertEquals(df.municipio_geocodigo.value_counts().size, 1)

    def test_alerta_all_munic(self):
        df = get_alerta_table()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(df.size, 0)
        self.assertGreater(df.municipio_geocodigo.value_counts().size, 1)

class TestSatellite(unittest.TestCase):
    def test_get_5d_image(self):
        fetcher = LandSurfaceTemperature()
        fetcher.get_5day_average_image(44.01104, 42.5841, 23.0134, 22.4916, start_date='20160101', end_date='20160106')
        files = glob('*.tiff')
        self.assertGreater(len(files), 0)
        os.system("rm *.tiff*")

    def test_alerta_state(self):
        df = get_alerta_table(state='RJ')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(df.size, 0)
        assert (df.municipio_geocodigo.values>3300000).all()
        assert (df.municipio_geocodigo.values < 4000000).all()

    def test_multi_city_dataset(self):
        df = build_multicity_dataset('RJ')
        self.assertGreater(len(df.columns), 500)


if __name__ == '__main__':
    unittest.main()
