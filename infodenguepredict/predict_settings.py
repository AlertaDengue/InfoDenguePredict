"""
This module contain the global parameter for the LSTM model
Created on 28/08/17
by fccoelho
license: GPL V3 or Later
"""
#======Global Model parameters=======
state = 'RJ'


#=======Clustering parameters========
# Variables to include in the correlation distance
cluster_vars = [
    "casos"
]
color_treshold = 0.6 # threshold for coloring the dendrogram

tmp_path = '/tmp'

#=======LSTM parameters==============

predictors = [
    "casos",
    "numero",
    "temp_min",
    "umid_min",

]
city = 3304557
