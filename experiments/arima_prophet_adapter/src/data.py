import fire
import pandas as pd
from neuralforecast.data.datasets.m3 import M3, M3Info
from neuralforecast.data.datasets.m4 import M4, M4Info
from neuralforecast.data.datasets.tourism import Tourism, TourismInfo


dict_datasets = {
    'Tourism': (Tourism, TourismInfo),
    'M3': (M3, M3Info),
    'M4': (M4, M4Info)
}

def get_data(directory: str, dataset: str, group: str, train: bool = True):
    if dataset not in dict_datasets.keys():
        raise Exception(f'dataset {dataset} not found')

    dataclass, datainfo = dict_datasets[dataset]
    if group not in datainfo.groups:
        raise Exception(f'group {group} not found for {dataset}')

    Y_df, *_ = dataclass.load(directory, group)

    horizon = datainfo[group].horizon
    freq = datainfo[group].freq
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
