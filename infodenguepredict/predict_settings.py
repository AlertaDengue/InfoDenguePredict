"""
This module contain the global parameter for the LSTM model
Created on 28/08/17
by fccoelho
license: GPL V3 or Later
"""
#======Global importing data parameters=======
STATE = 'RJ'

# Data_types: list of types of data to get into combined_data function
# Possible types: 'alerta', 'weather', 'tweet'
DATA_TYPES = ['alerta', 'weather']

#=======Clustering parameters========
# Variables to include in the correlation distance
CLUSTER_VARS = [
    "casos"
]

COLOR_THRESHOLD = 0.6 # threshold for coloring the dendrogram
TMP_PATH = '/tmp' #path to temporary files for clustering aux data


#=======LSTM parameters==============

# Predictors must change to fit data_types list
PREDICTORS = [
    'casos',
    'casos_est',
    'casos_est_min',
    'casos_est_max',
    'p_rt1',
    'p_inc100k',
    'temp_min',
    'temp_max',
    'umid_min',
    'pressao_min',
    # 'numero'
]

HIDDEN = 4
LOOK_BACK = 4
BATCH_SIZE = 1
PREDICTION_WINDOW = 3  # weeks
# city = 3304557 # Rio de Janeiro
# city = 3303500 # Nova Iguaçu
# city = 3301009 # Campos dos Goytacazes
# city = 3304904 # Sao Gonçalo
# city = 3303906 # Petropolis
# city = 3302858 # Mesquita
# city = 3303203 # Nilopolis
CITY = 3304144 # Queimados
# city = 3205309 # Vitoria
# city = 3205200 # Vila Velha, ES
EPOCHS = 100
