from itertools import product

import numpy as np
import pandas as pd
from datasetsforecast.m4 import M4Info 

from data import get_data

def evaluate_ws(lib: str, group: str): 
    
    horizon = M4Info[group].horizon 
    levels = np.arange(0.05, 0.5, 0.05)
    alpha = np.flip(levels)
    
    # Load forecast 
    try:
        forecast = pd.read_csv(f'/home/ubuntu/statsforecast/experiments/ces_intervals/data/{lib}-ces-forecasts-M4-{group}-pred-int.csv') 
        forecast = forecast.sort_values('unique_id')
        lowerb = forecast.loc[:, forecast.columns.str.contains('lowerb')].values.reshape(-1, len(levels)) 
        upperb = forecast.loc[:, forecast.columns.str.contains('upperb')].values.reshape(-1, len(levels))
    except:
        return None
    
    # Load actual data 
    try: 
        test_set = pd.read_csv(f'/home/ubuntu/statsforecast/experiments/ces_intervals/data/m4/datasets/{group}-test.csv') 
        test_set = test_set.sort_values('V1')
        test = test_set.drop(test_set.columns[[0]], axis = 1)
        test = test.values.reshape(-1, horizon) 
    except: 
        return None 
    
    ws = np.full(len(test), np.nan, np.float32) 
    for k in range(0, len(test)): 
        actual = test[k] 
        lo = lowerb[k*horizon:k*horizon+horizon]
        hi = upperb[k*horizon:k*horizon+horizon]
        wk = 0 
        for i in range(0, len(actual)): 
            for j in range(0, len(levels)): 
                if actual[i] < lo[i][j]: 
                    wj = (hi[i][j]-lo[i][j])+(2/alpha[j])*(lo[i][j]-actual[i])
                elif actual[i] > hi[i][j]: 
                    wj = (hi[i][j]-lo[i][j])+(2/alpha[j])*(actual[i]-hi[i][j])
                else:  
                    wj = hi[i][j]-lo[i][j]
                wk = wk+wj
        wk = wk/(len(actual)*len(levels))
        ws[k] = wk # Winkler score for series Y_k 
        
    times = pd.read_csv(f'/home/ubuntu/statsforecast/experiments/ces_intervals/data/{lib}-ces-time-M4-{group}-pred-int.csv')
    evals = times
    evals['Winkler-score (with mean)'] = ws.mean()
    evals['Winkler-score (with median)'] = np.median(ws)
    evals['dataset'] = group
    
    return evals

if __name__ == '__main__':
    groups = ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']
    lib = ['statsforecast']
    evaluation = [evaluate_ws(lib, group) for lib, group in product(lib, groups)]
    evaluation = [eval_ for eval_ in evaluation if eval_ is not None]
    evaluation = pd.concat(evaluation).sort_values(['dataset', 'model']).reset_index(drop=True)
    evaluation = evaluation[['dataset', 'model', 'Winkler-score (with mean)', 'Winkler-score (with median)', 'time']]
    #evaluation['time'] /= 60 #minutes
    evaluation = evaluation.set_index(['dataset', 'model']).stack().reset_index()
    evaluation.columns = ['dataset', 'model', 'metric', 'val']
    evaluation = evaluation.set_index(['dataset', 'metric', 'model']).unstack().round(2)
    evaluation = evaluation.droplevel(0, 1).reset_index()
    evaluation.to_csv('data/m4-evaluation.csv')
    print(evaluation.to_markdown(index=False))