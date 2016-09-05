"""
Tests related to data fetching code
"""

import unittest
from infodenguepredict.data.infodengue import get_temperature_data, get_alerta_table, get_tweet_data
import pandas as pd


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


if __name__ == '__main__':
    unittest.main()
