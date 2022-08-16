import fire
import pandas as pd
from neuralforecast.data.datasets.m3 import M3, M3Info
from neuralforecast.data.datasets.m4 import M4, M4Info
from neuralforecast.data.datasets.tourism import Tourism, TourismInfo
from neuralforecast.data.datasets.long_horizon import LongHorizon, LongHorizonInfo


dict_datasets = {
    'Tourism': (Tourism, TourismInfo),
    'M3': (M3, M3Info),
    'M4': (M4, M4Info),
    'LongHorizon': (LongHorizon, LongHorizonInfo)
}

def get_data(directory: str, dataset: str, group: str, train: bool = True):
    if dataset == 'ERCOT':
        data_location = "https://raw.githubusercontent.com/ourownstory/neuralprophet-data/main/datasets/"
        df_ercot = pd.read_csv(data_location + "multivariate/load_ercot_regions.csv")
        df_ercot_y = pd.read_csv(data_location + "energy/load_ercot.csv")
        df_ercot['y'] = df_ercot_y['y']
        Y_df = df_ercot.reset_index(drop=True)
        Y_df['unique_id'] = 'ERCOT'
        horizon = 24
        seasonality = 24
        freq = 'H'
        Y_df_test = Y_df.groupby('unique_id').tail(horizon)
        Y_df = Y_df.drop(Y_df_test.index)
        if train:
            return Y_df, horizon, freq, seasonality

        return Y_df_test, horizon, freq, seasonality

    if train:
        return Y_df, horizon, freq, seasonality


    if dataset not in dict_datasets.keys():
        raise Exception(f'dataset {dataset} not found')

    dataclass, datainfo = dict_datasets[dataset]
    if group not in datainfo.groups:
        raise Exception(f'group {group} not found for {dataset}')

    Y_df, *_ = dataclass.load(directory, group)

    if dataset == 'LongHorizon':
        horizon = datainfo[group].horizons[0]
    else:
        horizon = datainfo[group].horizon
    freq = datainfo[group].freq
    if group == 'ETTm2':
        seasonality = 24
    else:
        seasonality = datainfo[group].seasonality
    Y_df_test = Y_df.groupby('unique_id').tail(horizon)
    Y_df = Y_df.drop(Y_df_test.index)

    if train:
        return Y_df, horizon, freq, seasonality

    return Y_df_test, horizon, freq, seasonality

def save_data(dataset: str, group: str, train: bool = True):
    df, *_ = get_data('data', dataset, group, train)
    if train:
        df.to_csv(f'data/{dataset}-{group}.csv', index=False)
    else:
        df.to_csv(f'data/{dataset}-{group}-test.csv', index=False)


if __name__=="__main__":
    fire.Fire(save_data)
