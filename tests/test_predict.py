#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from infodenguepredict.predict import predict

class TestPredictionModel(unittest.TestCase):
    def test_predict(self):
        self.assertEqual(predict(None, None), None)


__author__ = "Flávio Codeço Coelho"
__copyright__ = "Flávio Codeço Coelho"
__license__ = "gpl3"



