import pandas as pd
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from infodenguepredict.data.infodengue import get_alerta_table


if __name__ == "__main__":
    data = get_alerta_table(3303609)  # Nova Iguaçu: 3303609
    tseries = importr('tseries')

    # pandas2ri.py2ri(data)


