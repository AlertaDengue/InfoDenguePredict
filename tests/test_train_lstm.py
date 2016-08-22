__author__ = 'fccoelho'

import unittest
from infodenguepredict.models import lstm
import pandas as pd

class TestDataPrep(unittest.TestCase):
    def test_get_data_all(self):
        df = lstm.get_example_table()
        self.assertIsInstance(df, pd.DataFrame)

    def test_get_data_one_city(self):
        df = lstm.get_example_table(3303609)
        self.assertIsInstance(df, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
