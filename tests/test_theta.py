import math

import numpy as np
import pytest
from statsforecast._lib import theta as _theta
from statsforecast.theta import (
    auto_theta,
    forecast_theta,
    forward_theta,
    initparamtheta,
    switch_theta,
    thetamodel,
)
from statsforecast.utils import AirPassengers as ap


@pytest.fixture
def theta_params():
    """Common theta model parameters."""
    return {"initial_smoothed": ap[0] / 2, "alpha": 0.5, "theta": 2.0}


@pytest.fixture
def init_theta_states(air_passengers, theta_params):
    """Initialize theta model states for all model types."""
    # Initialize all model types
    _theta.init_state(
        air_passengers,
        _theta.ModelType.STM,
        theta_params["initial_smoothed"],
        theta_params["alpha"],
        theta_params["theta"],
    )
    _theta.init_state(
        air_passengers,
        _theta.ModelType.OTM,
        theta_params["initial_smoothed"],
        theta_params["alpha"],
        theta_params["theta"],
    )
    _theta.init_state(
        air_passengers,
        _theta.ModelType.DSTM,
        theta_params["initial_smoothed"],
        theta_params["alpha"],
        theta_params["theta"],
    )
    _theta.init_state(
        air_passengers,
        _theta.ModelType.DOTM,
        theta_params["initial_smoothed"],
        theta_params["alpha"],
        theta_params["theta"],
    )
    return True


@pytest.fixture
def intermittent_series():
    """Test intermittent time series data."""
    return np.array([
        1., 0., 0., 1., 1., 1., 0., 0., 0., 1., 3., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 3., 0., 0., 0., 0., 0., 0.,
        0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 1., 1.,
        0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1.,
        0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 3., 1., 0., 1., 0., 0., 0.,
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2.,
        1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 1., 2., 0.,
        1., 0., 2., 2., 0., 0., 1., 2., 0., 0., 0., 2., 0., 1., 0., 0., 0.,
        0., 2., 0., 1., 0., 2., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 1.,
        0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 1., 1., 0.,
        0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 2.,
        1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 2., 0., 1., 0.,
        0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0.,
        1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 2., 0., 0., 0., 0., 1., 0., 1., 0., 2., 0., 0., 2., 0., 0.,
        2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 2.,
        0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
        1., 0., 1., 3., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 1., 0., 0., 2., 0., 0., 1., 0., 2., 0., 0., 0., 0.,
        2., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        1., 0., 1., 0., 0., 0., 0., 3., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 2., 0., 1., 0., 2., 1., 2., 2., 0., 0., 0., 0., 0., 0.,
        0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 2., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.,
        0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 2., 2.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 4., 0., 0., 0., 0., 0., 1.,
        1., 0., 0., 1., 1., 0., 0., 2., 1., 1., 1., 2., 1., 0., 0., 0., 1.,
        0., 0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
        0., 0., 0., 0., 0., 1., 2., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
        1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
        1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 2., 0., 0.,
        1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0.,
        0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 2., 0.,
        0., 0., 0., 2., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 2., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 2., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 1.,
        0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0.,
        0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.,
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
        1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        0., 0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
        0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 3., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 2., 0., 0., 1.,
        0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
        0., 0., 2., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    dtype=np.float32)  # fmt: skip


class TestThetaModel:
    """Test class for Theta model functionality."""

    def test_simple_theta_model(self, air_passengers, theta_params):
        """Test simple theta model calculations and fitted values."""
        amse_ = np.zeros(30)
        e_ = np.zeros(len(air_passengers))
        init_states = np.zeros((len(air_passengers), 5))

        mse = _theta.calc(
            air_passengers,
            init_states,
            _theta.ModelType.STM,
            theta_params["initial_smoothed"],
            theta_params["alpha"],
            theta_params["theta"],
            e_,
            amse_,
            3,
        )

        # verify we recover the fitted values
        np.testing.assert_array_equal(air_passengers - e_, init_states[:, -1])

        # verify we get same fitted values than R
        # use stm(AirPassengers, s=F, estimation=F, h = 12) to recover
        np.testing.assert_array_almost_equal(
            init_states[:, -1][[0, 1, -1]],
            np.array([101.1550, 107.9061, 449.1692]),
            decimal=2,
        )

        # recover mse
        assert math.isclose(np.sum(e_**2) / np.mean(np.abs(air_passengers)), mse)

    def test_simple_theta_forecasts(self, air_passengers, theta_params):
        """Test simple theta model forecasts."""
        # Setup states for STM
        amse_ = np.zeros(30)
        e_ = np.zeros(len(air_passengers))
        init_states = np.zeros((len(air_passengers), 5))

        _theta.calc(
            air_passengers,
            init_states,
            _theta.ModelType.STM,
            theta_params["initial_smoothed"],
            theta_params["alpha"],
            theta_params["theta"],
            e_,
            amse_,
            3,
        )

        # test forecasts
        h = 5
        fcsts = np.zeros(h)
        _theta.forecast(
            init_states,
            len(air_passengers),
            _theta.ModelType.STM,
            fcsts,
            theta_params["alpha"],
            theta_params["theta"],
        )

        # test same forecast than R's
        np.testing.assert_array_almost_equal(
            fcsts,
            np.array([441.9132, 443.2418, 444.5704, 445.8990, 447.2276]),
            decimal=3,
        )

    def test_optimal_theta_model(self, air_passengers, theta_params):
        """Test optimal theta model calculations and fitted values."""
        amse_ = np.zeros(30)
        e_ = np.zeros(len(air_passengers))
        init_states = np.zeros((len(air_passengers), 5))

        mse = _theta.calc(
            air_passengers,
            init_states,
            _theta.ModelType.OTM,
            theta_params["initial_smoothed"],
            theta_params["alpha"],
            theta_params["theta"],
            e_,
            amse_,
            3,
        )

        # verify we recover the fitted values
        np.testing.assert_array_equal(air_passengers - e_, init_states[:, -1])

        # verify we get same fitted values than R
        # use stm(AirPassengers, s=F, estimation=F, h = 12) to recover
        np.testing.assert_array_almost_equal(
            init_states[:, -1][[0, 1, -1]],
            np.array([101.1550, 107.9061, 449.1692]),
            decimal=2,
        )

        # recover mse
        assert math.isclose(np.sum(e_**2) / np.mean(np.abs(air_passengers)), mse)

    def test_optimal_theta_forecasts(self, air_passengers, theta_params):
        """Test optimal theta model forecasts."""
        # Setup states for OTM
        amse_ = np.zeros(30)
        e_ = np.zeros(len(air_passengers))
        init_states = np.zeros((len(air_passengers), 5))

        _theta.calc(
            air_passengers,
            init_states,
            _theta.ModelType.OTM,
            theta_params["initial_smoothed"],
            theta_params["alpha"],
            theta_params["theta"],
            e_,
            amse_,
            3,
        )

        # test forecasts
        h = 5
        fcsts = np.zeros(h)
        _theta.forecast(
            init_states,
            len(air_passengers),
            _theta.ModelType.OTM,
            fcsts,
            theta_params["alpha"],
            theta_params["theta"],
        )

        # test same forecast than R's
        np.testing.assert_array_almost_equal(
            fcsts,
            np.array([441.9132, 443.2418, 444.5704, 445.8990, 447.2276]),
            decimal=3,
        )

    @pytest.mark.parametrize(
        "model_type,expected_forecasts",
        [
            (_theta.ModelType.DSTM, [441.9132, 443.2330, 444.5484, 445.8594, 447.1659]),
            (_theta.ModelType.DOTM, [441.9132, 443.2330, 444.5484, 445.8594, 447.1659]),
        ],
    )
    def test_dynamic_theta_models(
        self, air_passengers, theta_params, model_type, expected_forecasts
    ):
        """Test dynamic theta models (DSTM and DOTM)."""
        amse_ = np.zeros(30)
        e_ = np.zeros(len(air_passengers))
        init_states = np.zeros((len(air_passengers), 5))

        mse = _theta.calc(
            air_passengers,
            init_states,
            model_type,
            theta_params["initial_smoothed"],
            theta_params["alpha"],
            theta_params["theta"],
            e_,
            amse_,
            3,
        )

        # verify we recover the fitted values
        np.testing.assert_array_equal(air_passengers - e_, init_states[:, -1])

        # verify we get same fitted values than R
        # use dstm(AirPassengers, s=F, estimation=F, h = 12) to recover
        expected_fitted = [112.0000, 112.0000, 449.1805]
        np.testing.assert_array_almost_equal(
            init_states[:, -1][[0, 1, -1]], np.array(expected_fitted), decimal=2
        )

        # recover mse
        assert math.isclose(np.sum(e_[2:] ** 2) / np.mean(np.abs(air_passengers)), mse)

        # test forecasts
        h = 5
        fcsts = np.zeros(h)
        _theta.forecast(
            init_states,
            len(air_passengers),
            model_type,
            fcsts,
            theta_params["alpha"],
            theta_params["theta"],
        )

        # test same forecast than R's
        np.testing.assert_array_almost_equal(
            fcsts, np.array(expected_forecasts), decimal=3
        )

    def test_zero_constant_series(self):
        """Test theta model with zero constant time series."""
        zeros = np.zeros(30, dtype=np.float32)
        res = auto_theta(zeros, m=12)
        forecast_theta(res, 28)

    @pytest.mark.parametrize(
        "m,model,expected_mean,decimal_precision",
        [
            # Simple Theta Model with no seasonality
            (1, "STM", [432.9292, 434.2578, 435.5864, 436.9150, 438.2435], 2),
            # Simple Theta Model with seasonality
            (12, "STM", [440.7886, 429.0739, 490.4933, 476.4663, 480.4363], 0),
            # Optimized Theta Model with no seasonality - adjusted precision due to algorithm differences
            (
                1,
                "OTM",
                [433.72150701, 435.83452245, 437.94753789, 440.06055334, 442.17356878],
                0,
            ),
            # Optimized Theta Model with seasonality
            (12, "OTM", [442.8492, 432.1255, 495.1706, 482.1585, 487.3280], 0),
            # Dynamic Simple Theta Model with no seasonality
            (1, "DSTM", [432.9292, 434.2520, 435.5693, 436.8809, 438.1871], 2),
            # Dynamic Simple Theta Model with seasonality
            (12, "DSTM", [440.7631, 429.0512, 490.4711, 476.4495, 480.4251], 0),
            # Dynamic Optimized Theta Model with no seasonality - adjusted precision due to algorithm differences
            (1, "DOTM", [432.5131, 433.4257, 434.3344, 435.2391, 436.1399], 1),
            # Dynamic Optimized Theta Model with seasonality
            (12, "DOTM", [442.9720, 432.3586, 495.5702, 482.6789, 487.9888], 0),
        ],
    )
    def test_auto_theta_models(
        self, air_passengers, m, model, expected_mean, decimal_precision
    ):
        """Test auto_theta with different model types and seasonality."""
        res = auto_theta(air_passengers, m=m, model=model)
        fcst = forecast_theta(res, 5)

        np.testing.assert_almost_equal(
            expected_mean,
            fcst["mean"],
            decimal=decimal_precision,
        )

    def test_intermittent_series_forecasting(self, intermittent_series):
        """Test theta model with intermittent time series."""
        for season_length in [1, 7]:
            res = auto_theta(intermittent_series, m=season_length)
            fcst = forecast_theta(res, 28)

            # Basic sanity checks
            assert len(fcst["mean"]) == 28
            assert np.all(fcst["mean"] >= 0)  # Non-negative forecasts for count data

    def test_forward_theta_consistency(self, air_passengers):
        """Test that forward_theta maintains consistency."""
        res = auto_theta(air_passengers, m=12)

        # Test that forward_theta gives same results as original
        np.testing.assert_allclose(
            forecast_theta(forward_theta(res, air_passengers), h=12)["mean"],
            forecast_theta(res, h=12)["mean"],
        )

    def test_theta_initialization_functions(self, air_passengers):
        """Test theta initialization and parameter functions."""
        # Test parameter initialization
        initparamtheta(
            initial_smoothed=np.nan,
            alpha=np.nan,
            theta=np.nan,
            y=air_passengers,
            modeltype=_theta.ModelType.DOTM,
        )

        # Test model switching
        switch_theta("STM")

        # Test theta model with various parameters
        res = thetamodel(
            y=air_passengers,
            m=12,
            modeltype="STM",
            initial_smoothed=np.nan,
            alpha=np.nan,
            theta=np.nan,
            nmse=3,
        )

        # Test forecasting with confidence intervals
        fcst = forecast_theta(res, 12, level=[90, 80])
        assert "mean" in fcst
        assert "lo-90" in fcst
        assert "hi-90" in fcst
        assert "lo-80" in fcst
        assert "hi-80" in fcst

    def test_theta_with_additive_decomposition(self, air_passengers):
        """Test theta model with additive decomposition."""
        res = auto_theta(
            air_passengers, m=12, model="DOTM", decomposition_type="additive"
        )
        fcst = forecast_theta(res, 12, level=[80, 90])

        # Basic validation
        assert len(fcst["mean"]) == 12
        assert "lo-80" in fcst
        assert "hi-80" in fcst
        assert "lo-90" in fcst
        assert "hi-90" in fcst

    def test_transfer_functionality(self, air_passengers, intermittent_series):
        """Test transfer functionality between different series."""
        res = auto_theta(air_passengers, m=12)

        # Test transfer to intermittent series
        fcst = forecast_theta(
            forward_theta(res, intermittent_series), h=12, level=[80, 90]
        )
        assert len(fcst["mean"]) == 12

        # Test that parameters are preserved during transfer
        res_transfer = forward_theta(res, intermittent_series)
        for key in res_transfer["par"]:
            assert res["par"][key] == res_transfer["par"][key]


@pytest.mark.parametrize(
    "model,expected_mean",
    [
        ("STM", [807.9822, 806.6251, 805.2679]),
        ("OTM", [805.5509, 802.8423, 800.1336]),
        ("DSTM", [795.4948, 794.1524, 792.8146]),
        ("DOTM", [804.9958, 804.9958, 804.9958]),
    ],
)
def test_theta_estimation_matches_forectheta(model, expected_mean):
    """The fitted models should match forecTheta on a series where alpha < 0.99.

    The static models use all errors in the objective and the dynamic ones skip
    the first two. Recover with e.g. stm(Nile, h = 3) in forecTheta.
    """
    # Nile from R's datasets package
    nile = np.array(
        [
            1120, 1160, 963, 1210, 1160, 1160, 813, 1230, 1370, 1140, 995, 935,
            1110, 994, 1020, 960, 1180, 799, 958, 1140, 1100, 1210, 1150, 1250,
            1260, 1220, 1030, 1100, 774, 840, 874, 694, 940, 833, 701, 916,
            692, 1020, 1050, 969, 831, 726, 456, 824, 702, 1120, 1100, 832,
            764, 821, 768, 845, 864, 862, 698, 845, 744, 796, 1040, 759,
            781, 865, 845, 944, 984, 897, 822, 1010, 771, 676, 649, 846,
            812, 742, 801, 1040, 860, 874, 848, 890, 744, 749, 838, 1050,
            918, 986, 797, 923, 975, 815, 1020, 906, 901, 1170, 912, 746,
            919, 718, 714, 740
        ],
        dtype=np.float64,
    )
    res = auto_theta(nile, m=1, model=model)
    np.testing.assert_allclose(forecast_theta(res, 3)["mean"], expected_mean, rtol=2e-3)


def test_theta_short_series_no_nan_intervals():
    """Regression test for #1135: theta should not produce NaN intervals for short series."""
    import numpy as np
    from statsforecast.models import AutoTheta
    
    y = np.array([10.0, 12.0, 11.0, 13.0])
    model = AutoTheta(season_length=1)
    out = model.forecast(y=y, h=3, level=[95])
    
    assert not np.isnan(out["lo-95"]).any(), "NaN in lower prediction interval"
    assert not np.isnan(out["hi-95"]).any(), "NaN in upper prediction interval"
    assert not np.isnan(out["mean"]).any(), "NaN in point forecast"
