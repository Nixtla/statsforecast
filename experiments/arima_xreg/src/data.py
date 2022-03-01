import fire
import pandas as pd
from neuralforecast.data.datasets.epf import EPF


def get_data(directory: str, group: str, train: bool = True):

    Y_df, X_df, _ = EPF.load(directory, group)
    Y_df = Y_df.merge(X_df, how='left', on=['unique_id', 'ds'])
    Y_df = Y_df.drop(['day_0', 'week_day'], 1)

    horizon = 24 * 7
    Y_df = Y_df.groupby('unique_id').tail(7 * horizon).reset_index(drop=True)
    Y_df_test = Y_df.groupby('unique_id').tail(horizon)
    Y_df = Y_df.drop(Y_df_test.index)

    if train:
        return Y_df

    return Y_df_test

def save_data(group: str, train: bool = True):
    df = get_data('data', group, train)
    if train:
        df.to_csv(f'data/EPF-{group}.csv', index=False)
    else:
        df.to_csv(f'data/EPF-{group}-test.csv', index=False)


if __name__=="__main__":
    fire.Fire(save_data)
