__author__ = 'fccoelho'

import unittest

import pandas as pd

from infodenguepredict.models.deeplearning import lstm


class TestDataPrep(unittest.TestCase):
    def test_get_data_all(self):
        df = lstm.get_example_table()
        self.assertIsInstance(df, pd.DataFrame)

    def test_get_data_one_city(self):
        df = lstm.get_example_table(3303609)
        self.assertIsInstance(df, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
