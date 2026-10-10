import warnings
import numpy as np
import pandas as pd
import pytest
from statsforecast.adapters.prophet import Prophet, AutoARIMAProphet

warnings.simplefilter(action="ignore", category=FutureWarning)


@pytest.fixture
def prophet_model():
    """Create a Prophet model instance."""
    return Prophet(daily_seasonality=False)


@pytest.fixture
def autoarimaprophet_fitted_model():
    model = AutoARIMAProphet()

    history = pd.DataFrame({
        "ds": pd.date_range("2026-01-01", periods=30, freq="D"),
        "y": np.sin(np.arange(30) / 5) + np.arange(30) * 0.1,
    })

    model.fit(history, disable_seasonal_features=False)
    return model

@pytest.fixture
def holidays_data():
    """Create sample holidays data."""
    playoffs = pd.DataFrame(
        {
            "holiday": "playoff",
            "ds": pd.to_datetime(
                [
                    "2008-01-13",
                    "2009-01-03",
                    "2010-01-16",
                    "2010-01-24",
                    "2010-02-07",
                    "2011-01-08",
                    "2013-01-12",
                    "2014-01-12",
                    "2014-01-19",
                    "2014-02-02",
                    "2015-01-11",
                    "2016-01-17",
                    "2016-01-24",
                    "2016-02-07",
                ]
            ),
            "lower_window": 0,
            "upper_window": 1,
        }
    )
    superbowls = pd.DataFrame(
        {
            "holiday": "superbowl",
            "ds": pd.to_datetime(["2010-02-07", "2014-02-02", "2016-02-07"]),
            "lower_window": 0,
            "upper_window": 1,
        }
    )
    return pd.concat((playoffs, superbowls))


def test_prophet_initialization():
    """Test Prophet model initialization."""
    model = Prophet(daily_seasonality=False)
    assert model is not None
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_prophet_fit(sample_data_prophet, prophet_model):
    """Test Prophet model fitting."""
    prophet_model.fit(sample_data_prophet)
    # Check that the model has been fitted by verifying it has the necessary attributes
    assert hasattr(prophet_model, "history")
    assert prophet_model.history is not None


def test_prophet_predict(sample_data_prophet, prophet_model):
    """Test Prophet model prediction."""
    prophet_model.fit(sample_data_prophet)
    future = prophet_model.make_future_dataframe(365)
    forecast = prophet_model.predict(future)

    assert forecast is not None
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) > len(sample_data_prophet)


def test_prophet_make_future_dataframe(sample_data_prophet, prophet_model):
    """Test making future dataframe."""
    prophet_model.fit(sample_data_prophet)
    future = prophet_model.make_future_dataframe(365)

    assert isinstance(future, pd.DataFrame)
    assert len(future) == len(sample_data_prophet) + 365
    assert "ds" in future.columns


def test_prophet_plot(sample_data_prophet, prophet_model):
    """Test Prophet plotting functionality."""
    prophet_model.fit(sample_data_prophet)
    future = prophet_model.make_future_dataframe(365)
    forecast = prophet_model.predict(future)

    # Test that plot method exists and can be called
    fig = prophet_model.plot(forecast)
    assert fig is not None


def test_prophet_with_holidays(sample_data_prophet, holidays_data):
    """Test Prophet model with holidays."""
    model = Prophet(holidays=holidays_data, daily_seasonality=False)
    model.fit(sample_data_prophet)
    future = model.make_future_dataframe(365)
    forecast = model.predict(future)

    assert forecast is not None
    assert isinstance(forecast, pd.DataFrame)


def test_prophet_forecast_components(sample_data_prophet, prophet_model):
    """Test Prophet forecast components."""
    prophet_model.fit(sample_data_prophet)
    future = prophet_model.make_future_dataframe(10)
    forecast = prophet_model.predict(future)

    # Check that the forecast contains expected columns
    expected_columns = ["yhat", "yhat_lower", "yhat_upper", "trend"]
    for col in expected_columns:
        assert col in forecast.columns, f"Expected column '{col}' not found in forecast"


def test_prophet_empty_dataframe():
    """Test Prophet behavior with empty dataframe."""
    model = Prophet(daily_seasonality=False)
    empty_df = pd.DataFrame(columns=["ds", "y"])

    with pytest.raises(Exception):  # Prophet should raise an error with empty data
        model.fit(empty_df)

@pytest.mark.parametrize(
    "date_selection",
    [
        "full_training_and_test",
        "training_only",
        "partial_training_aligned_and_test",
        "partial_training_not_aligned_and_test",
        "test_only",
        "none",
    ],
)
def test_predict_with_partial_training_and_test_dates(
    autoarimaprophet_fitted_model, date_selection
):
    """Predictions should align correctly for training and future dates."""
    model = autoarimaprophet_fitted_model
    history = model.history

    last_date = history["ds"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=2,
        freq="D",
    )

    if date_selection == "full_training_and_test":
        train_dates = history["ds"].tolist()
        requested_dates = train_dates + list(future_dates)
        df = pd.DataFrame({"ds": requested_dates})

    elif date_selection == "training_only":
        df = history[["ds"]].copy()

    elif date_selection == "partial_training_aligned_and_test":
        train_dates = history["ds"].iloc[-5:].tolist()
        requested_dates = train_dates + list(future_dates)
        df = pd.DataFrame({"ds": requested_dates})

    elif date_selection == "partial_training_not_aligned_and_test":
        # Select non-consecutive training dates.
        train_dates = history["ds"].iloc[[0, 2, 5, 8, 10]].tolist()
        requested_dates = train_dates + list(future_dates)
        df = pd.DataFrame({"ds": requested_dates})

    elif date_selection == "test_only":
        df = pd.DataFrame({"ds": future_dates})

    else:  # none
        df = None

    result = model.predict(df)

    if date_selection == "none":
        expected_dates = history["ds"].reset_index(drop=True)
    else:
        expected_dates = df["ds"].reset_index(drop=True)

    # Correct shape and schema.
    assert len(result) == len(expected_dates)
    assert list(result.columns) == [
        "ds", "yhat", "yhat_lower", "yhat_upper"
    ]

    # Preserve the requested date order.
    pd.testing.assert_series_equal(
        result["ds"].reset_index(drop=True),
        expected_dates,
        check_names=False,
    )

    # All requested dates should have valid predictions.
    assert result["yhat"].notna().all()
    assert result["yhat_lower"].notna().all()
    assert result["yhat_upper"].notna().all()

    # Prediction intervals should contain the point forecasts.
    assert (result["yhat_lower"] <= result["yhat"]).all()
    assert (result["yhat"] <= result["yhat_upper"]).all()
