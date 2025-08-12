import warnings

import pandas as pd
import pytest
from statsforecast.adapters.prophet import Prophet

warnings.simplefilter(action="ignore", category=FutureWarning)


@pytest.fixture
def prophet_model():
    """Create a Prophet model instance."""
    return Prophet(daily_seasonality=False)


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
