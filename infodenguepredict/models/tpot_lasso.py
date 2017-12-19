import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from infodenguepredict.data.infodengue import get_cluster_data
from infodenguepredict.predict_settings import PREDICTORS, DATA_TYPES, STATE
import pickle
import matplotlib.pyplot as plt

def build_lagged_features(dt, lag=2, dropna=True):
    '''
    Builds a new DataFrame to facilitate regressing over all possible lagged features
    :param dt: Dataframe containing features
    :param lag: maximum lags to compute
    :param dropna: if true the initial rows containing NANs due to lagging will be dropped
    :return: Dataframe
    '''
    if type(dt) is pd.DataFrame:
        new_dict = {}
        for col_name in dt:
            new_dict[col_name] = dt[col_name]
            # create lagged Series
            for l in range(1, lag + 1):
                new_dict['%s_lag%d' % (col_name, l)] = dt[col_name].shift(l)
        res = pd.DataFrame(new_dict, index=dt.index)

    elif type(dt) is pd.Series:
        the_range = range(lag + 1)
        res = pd.concat([dt.shift(i) for i in the_range], axis=1)
        res.columns = ['lag_%d' % i for i in the_range]
    else:
        print('Only works for DataFrame or Series')
        return None
    if dropna:
        return res.dropna()
    else:
        return res

def plot_prediction(Xdata, ydata, model, title):
    plt.figure()
    preds = model.predict(Xdata)
    plt.plot(preds, ':', label='Tpot')
    plt.plot(ydata, alpha=0.3, label='Data')
    plt.legend(loc=0)
    plt.title(title)
    plt.savefig('TPOT{}_{}.png'.format(city, title))
    plt.show()

# NOTE: Make sure that the class is labeled 'target' in the data file

if __name__=="__main__":
    lookback = 12
    horizon = 10  # weeks
    city = 3303302
    target = 'casos_{}'.format(city)
    with open('infodenguepredict/analysis/clusters_{}.pkl'.format(STATE), 'rb') as fp:
        clusters = pickle.load(fp)
    data, group = get_cluster_data(city, clusters=clusters, data_types=DATA_TYPES, cols=PREDICTORS)

    data_lag = build_lagged_features(data, lookback)
    data_lag.dropna()
    targets = {}
    for d in range(1, horizon+1):
        targets[d] = data_lag[target].shift(-d)[:-horizon]

    X_train, X_test, y_train, y_test = train_test_split(data_lag, data_lag[target],
                                                        train_size=0.75, test_size=0.25, shuffle=False)
    X_test = X_test.iloc[:-horizon]

    X_train[target].plot()
    targets[2].plot(label='target')
    plt.legend(loc=0)
    plt.show()



    for d in range(1, horizon+1):
        exported_pipeline = make_pipeline(
            VarianceThreshold(threshold=0.45),
            LassoLarsCV(normalize=False)
        )
        tgt = targets[d][:len(X_train)]
        tgtt = targets[d][len(X_train):]

        exported_pipeline.fit(X_train, tgt)

    # results = exported_pipeline.predict(testing_features)
    #     plot_prediction(X_train.values, tgt.values, exported_pipeline, 'In_Sample_{}'.format(d))
        plot_prediction(X_test.values, tgtt.values, exported_pipeline, 'Out_of_Sample_{}'.format(d))

