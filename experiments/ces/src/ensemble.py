import fire
import pandas as pd


def main(dataset: str, group: str):
    ets = pd.read_csv(f'./data/ets-forecasts-{dataset}-{group}.csv')
    ces = pd.read_csv(f'./data/ces-forecasts-{dataset}-{group}.csv')
    df = ets.merge(ces, how='left', on=['unique_id', 'ds'])
    df['ensemble'] = df[['ets', 'ces']].mean(axis=1)
    df = df[['unique_id', 'ds', 'ensemble']]
    df.to_csv(f'./data/ensemble-forecasts-{dataset}-{group}.csv', index=False)

    # time
    ets = pd.read_csv(f'./data/ets-time-{dataset}-{group}.csv')
    ces = pd.read_csv(f'./data/ces-time-{dataset}-{group}.csv')
    df = pd.DataFrame({'time': ets['time'] + ces['time'], 'model': 'ensemble'}, index=[0])
    df.to_csv(f'./data/ensemble-time-{dataset}-{group}.csv', index=False)

if __name__=="__main__":
    fire.Fire(main)
