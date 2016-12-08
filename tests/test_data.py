"""
Tests related to data fetching code
"""

import unittest
from infodenguepredict.data.infodengue import get_temperature_data, get_alerta_table, get_tweet_data
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

    def test_alerta(self):
        df = get_alerta_table(3303609)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(df.size, 0)

class TestSatellite(unittest.TestCase):
    def test_get_5d_image(self):
        fetcher = LandSurfaceTemperature()
        fetcher.get_5day_average_image(44.01104, 42.5841, 23.0134, 22.4916, start_date='20160101', end_date='20160106')
        files = glob('*.tiff')
        self.assertGreater(len(files), 0)
        os.system("rm *.tiff*")



if __name__ == '__main__':
    unittest.main()
