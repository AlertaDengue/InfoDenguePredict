import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
import matplotlib.dates as mdates
import sys
sys.path.append('../../')
from infodenguepredict.data.infodengue import get_cases_table

def get_data(city, start_train = '2016-01-01', end_train = '2021-12-31'): 
    '''
    Load data and split into train and test 
    '''
    df = get_cases_table(city).rename(columns = {'casos':'y'})
    #df.set_index('ds', inplace = True)
    df.index = pd.to_datetime(df.index)

    df.y = np.log(df.y)

    df = df.fillna(0)

    df.replace([np.inf, -np.inf], 0, inplace=True)

    df_train = df.loc[start_train:end_train]

    df_test = df.loc[end_train:]

    return df_train, df_test 

def tune_model(df_train): 

    model=auto_arima(df_train, 
                    start_p=6,
                    start_q=6,
                    start_d=2,
                    max_p=12,
                    max_q=12,
                    max_d=5,
                    max_order=12,
                    seasonal=False,
                    trace=True,
                    maxiter = 100, 
                    error_action='ignore',
                    information_criterion = 'aic',suppress_warnings=True,
                    stepwise=True)
    
    model.fit(df_train)

    return model 

def plot_in_sample(ax1, ax2, model, df_train):

    df_in_sample = pd.DataFrame()

    preds_in_sample = model.predict_in_sample(return_conf_int=True)

    df_in_sample['date'] = df_train.index

    df_in_sample[['lower', 'upper']] = preds_in_sample[1]

    df_in_sample['pred'] = preds_in_sample[0]

    df_in_sample = df_in_sample.iloc[1:]

    ax1.plot(df_in_sample.date, df_in_sample.pred, color = 'tab:orange', label = 'ARIMA')

    ax1.fill_between(df_in_sample.date, df_in_sample.lower, df_in_sample.upper, color = 'tab:orange', alpha = 0.3)

    ax1.plot(df_train.y, color = 'black', label = 'Data')

    ax1.legend()

    ax1.grid()

    ax1.set_title('Train - log')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%y'))

    ax2.plot(df_in_sample.date, np.exp(df_in_sample.pred), color = 'tab:orange', label = 'ARIMA')

    ax2.fill_between(df_in_sample.date, np.exp(df_in_sample.lower), np.exp(df_in_sample.upper), color = 'tab:orange', alpha = 0.3)

    ax2.plot(np.exp(df_train.y), color = 'black', label = 'Data')

    ax2.legend()

    ax2.grid()

    ax2.set_title('Train')

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%y'))


def rolling_window(model, df, steps = 4):
    '''
    Rolling fashion to eval the performance of the model out of sample predicting 4 weeks ahead.
            '''
    
    preds = []
    preds_lower = []
    preds_upper = [] 
    
    for n in np.arange(0, df.shape[0]-steps):
        
        p = model.update(df.iloc[n]).predict(steps,return_conf_int = True)
        
        preds.append(p[0][-1])
        preds_lower.append(p[1][:,0][-1])
        preds_upper.append(p[1][:,1][-1])
        
    df_preds = pd.DataFrame()
    
    df_preds['dates'] = df.index[steps:]
    
    df_preds['preds_lower'] = preds_lower
    
    df_preds['preds'] = preds
    
    df_preds['preds_upper'] = preds_upper
        
    return df_preds

def plot_out_of_sample(ax3, ax4, model, df_test): 
    
    df_preds = rolling_window(model, df_test)

    ax3.plot(df_preds.dates, df_preds.preds, color = 'tab:orange', label = 'ARIMA')

    ax3.fill_between(df_preds.dates, df_preds.preds_lower, df_preds.preds_upper, color = 'tab:orange', alpha = 0.3)

    ax3.plot(df_test.y, color = 'black', label = 'Data')

    ax3.legend()

    ax3.grid()

    ax3.set_title('Test - log')

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%y'))

    ax4.plot(df_preds.dates, np.exp(df_preds.preds), color = 'tab:orange', label = 'ARIMA')

    ax4.fill_between(df_preds.dates, np.exp(df_preds.preds_lower), np.exp(df_preds.preds_upper), color = 'tab:orange', alpha = 0.3)

    ax4.plot(np.exp(df_test.y), color = 'black', label = 'Data')

    ax4.legend()

    ax4.grid()

    ax4.set_title('Test')

    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%y'))


def plot_train_test(city): 

    df_train, df_test = get_data(city)

    model = tune_model(df_train) 

    fig,ax = plt.subplots(2,2, figsize = (12,8))

    plot_in_sample(ax[0,0], ax[0,1], model, df_train)

    plot_out_of_sample(ax[1,0], ax[1,1], model, df_test)


    fig.suptitle(f'ARIMA model in {city} - 4 steps ahead', fontsize = 14)

    plt.savefig(f'../results/val_arima_{city}', bbox_inches = 'tight', dpi = 300)

    plt.close()

capitais = [4314902,
            4205407,
            4106902, 
            3550308,
            3304557,
            3106200,
            3205309,
            5103403,
            5002704,
            5208707,
            5300108, 
            2704302, # Al
            2927408, # BA 
            2304400, # CE
            2111300, #MA
            2507507, #PB
            2611606, #PE
            2211001, #PI
            2800308, #SE
            2408102, #RN
            1200401, #AC
            1302603, #AM
            1600303, #AP
            1721000, #TO
            1100205, #RO
            1400100, #RR
            1501402, #PA
           ]
if __name__ == "__main__":
    for city in capitais:
        print(f'Training model at {city}')
        plot_train_test(city)









