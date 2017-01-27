u"""
Created on 27/01/17
by fccoelho
license: GPL V3 or Later

adapted from this example:
http://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as P
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, classification_report
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten


def split_data(df, train_fraction=0.6):
    train_size = int(len(df) * train_fraction)
    test_size = len(df) - train_size
    train, test = df[0:train_size, :], df[train_size:len(df), :]
    print(len(train), len(test))
    return train,test
