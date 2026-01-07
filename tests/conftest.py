import pandas as pd
import pytest
from statsforecast.utils import AirPassengers as ap


@pytest.fixture(scope="module")
def sample_data_prophet():
    """Load sample data for Prophet testing."""
    df = pd.read_csv(
        "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
    )
    return df


@pytest.fixture
def air_passengers():
    """AirPassengers dataset."""
    return ap
