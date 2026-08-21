import numpy as np
import pandas as pd
from statsforecast.mfles import MFLES
from statsforecast.models import MFLES as ModelMFLES


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
        cls.timeseries = timeseries.set_index("ds").resample("H").sum()

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


def test_mfles_multiseasonal_forecast_uses_full_cycle():
    season_1 = np.array([0, 100])
    season_2 = np.array([0, -200, 0, 0, 0])
    y = np.resize(season_1, 30) + np.resize(season_2, 30)

    model = ModelMFLES(season_length=[2, 5], trend_lr=0, residuals_lr=0)
    model.fit(y)
    forecast = model.predict(10)["mean"]

    # The combined seasonality repeats every lcm(2, 5) observations.
    np.testing.assert_allclose(forecast[5] - forecast[0], 100, atol=1)
