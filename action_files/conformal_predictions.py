import pytest
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ZeroModel, Naive
from statsforecast.utils import ConformalIntervals
import matplotlib.pyplot as plt



@pytest.mark.parametrize("unit_prediction_step", [True, False])
def test_unit_prediction_step(unit_prediction_step):
    df = pd.read_parquet('https://datasets-nixtla.s3.amazonaws.com/m4-hourly.parquet')
    df = df.iloc[:748, :]

    h = 24
    steps = 1 if unit_prediction_step else h
    intervals = ConformalIntervals(h=h, n_windows=int((len(df) - 1) / h), method='naive_error')
    sf = StatsForecast(models=[Naive(prediction_intervals=intervals, step=steps)], freq=1, n_jobs=1)
    sf.fit(df=df)
    preds = sf.predict(h=h, level=[50])
    preds.iloc[:, 2:].plot()
    plt.show()

