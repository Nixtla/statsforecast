import pandas as pd
import pytest
from statsforecast.utils import AirPassengers as ap


@pytest.fixture(scope="session", autouse=True)
def setup_ray():
    """Initialize Ray with low resource configuration for testing."""
    try:
        import ray
    except ImportError:
        # Ray not installed, skip setup
        yield
        return

    # Initialize Ray with limited resources to avoid memory issues
    # Disable working_dir upload for local testing to avoid package size issues
    ray.init(
        num_cpus=2,
        object_store_memory=500 * 1024 * 1024,  # 500 MB
        _memory=1024 * 1024 * 1024,  # 1 GB total memory
        ignore_reinit_error=True,
        runtime_env={
            "working_dir": None,  # Don't upload working directory for local testing
        },
    )

    yield

    # Shutdown Ray after all tests
    ray.shutdown()


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
