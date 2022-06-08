from pathlib import Path

from statsforecast.utils import generate_series


def main():
    dir_ = Path('./data')
    dir_.mkdir(exist_ok=True)
    for length in [10_000, 100_000, 1_000_000, 5_000_000, 10_000_000]:
        print(f'Length: {length}')
        file_ = dir_ / f'series_{length}.parquet'
        if file_.exists():
            print('Already generated')
            continue
        series = generate_series(n_series=length, seed=1, 
                                 min_length=20, 
                                 max_length=100, 
                                 equal_ends=True)
        series.to_parquet(file_)
        print('Data saved')

if __name__=="__main__":
    main()
