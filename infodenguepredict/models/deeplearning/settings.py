"""
This module contain the global parameter for the LSTM model
Created on 28/08/17
by fccoelho
license: GPL V3 or Later
"""
#=======Clustering parameters========
# Variables to include in the correlation distance
cluster_vars = [
    "casos",
    "numero"
]


#=======LSTM parameters==============

predictors = [
    "casos",
    "numero",
    "temp_min",
    "umid_min",

]
