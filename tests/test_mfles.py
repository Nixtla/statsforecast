import numpy as np
import pandas as pd
from statsforecast.mfles import MFLES


class TestMFLES:
    @classmethod
    def setup_class(cls):
        url = "https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/VIC2015/demand.csv"
        df = pd.read_csv(url)
        df["Date"] = df["Date"].apply(
            lambda x: pd.Timestamp("1899-12-30") + pd.Timedelta(x, unit="days")
        )
        df["ds"] = df["Date"] + pd.to_timedelta((df["Period"] - 1) * 30, unit="m")
        timeseries = df[["ds", "OperationalLessIndustrial"]]
        timeseries.columns = [
            "ds",
            "y",
        ]  # Rename to OperationalLessIndustrial to y for simplicity.

        # Filter for first 149 days of 2012.
        start_date = pd.to_datetime("2012-01-01")
        end_date = start_date + pd.Timedelta("149D")
        mask = (timeseries["ds"] >= start_date) & (timeseries["ds"] < end_date)
        timeseries = timeseries[mask]

        # Resample to hourly
        cls.timeseries = timeseries.set_index("ds").resample("h").sum()

    def test_mfles_fit_predict(self):
        mfles = MFLES()
        fitted = mfles.fit(y=self.timeseries.y.values, seasonal_period=[24, 24 * 7])
        predicted = mfles.predict(forecast_horizon=24)

        assert fitted is not None
        assert predicted is not None
        assert len(predicted) == 24

    def test_mfles_optimize(self):
        mfles = MFLES()
        opt_params = mfles.optimize(
            self.timeseries.y.values,
            seasonal_period=[24, 24 * 7],
            n_steps=3,
            test_size=24,
            step_size=24,
        )
        fitted = mfles.fit(y=self.timeseries.y.values, seasonal_period=[24, 24 * 7])
        predicted = mfles.predict(forecast_horizon=24)

        assert opt_params is not None
        assert fitted is not None
        assert predicted is not None
        assert len(predicted) == 24


def test_mfles_seasonal_period_longer_than_series():
    model = MFLES()
    model.fit(np.arange(8, dtype=float), seasonal_period=12)

    forecast = model.predict(12)
    assert np.isfinite(forecast).all()


def test_mfles_multiple_seasonalities_forecast():
    periods = [2, 3, 5]
    n_obs = 151
    h = 2 * np.lcm.reduce(periods)
    patterns = [[0, 100], [0, -150, 50], [0, -200, 0, 0, 0]]
    y_full = sum(np.resize(p, n_obs + h) for p in patterns).astype(float)
    y, y_future = y_full[:n_obs], y_full[n_obs:]

    model = MFLES()
    model.fit(
        y, seasonal_period=periods, seasonal_lr=1, linear_lr=0, rs_lr=0
    )
    forecast = model.predict(h)

    np.testing.assert_allclose(forecast, y_future, atol=1e-2)
    assert [len(s) for s in model.seasonality] == periods

    # Emulate the combined seasonal array stored by previously fitted MFLES models.
    legacy_h = max(periods)
    model.seasonality = sum(np.resize(s, legacy_h) for s in model.seasonality)
    np.testing.assert_allclose(model.predict(legacy_h), forecast[:legacy_h])
