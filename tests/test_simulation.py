import numpy as np
import pandas as pd
import pytest
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoARIMA, AutoCES, AutoTheta


def test_simulate_basic():
    df = pd.DataFrame(
        {
            "unique_id": [1] * 20 + [2] * 20,
            "ds": pd.to_datetime(["2020-01-01"] * 40)
            + pd.to_timedelta(np.tile(np.arange(20), 2), unit="D"),
            "y": np.concatenate(
                [
                    np.arange(20) + np.random.normal(0, 0.1, 20),
                    np.arange(20, 0, -1) + np.random.normal(0, 0.1, 20),
                ]
            ),
        }
    )

    models = [
        AutoETS(season_length=1),
        AutoARIMA(season_length=1),
        AutoCES(season_length=1),
        AutoTheta(season_length=1),
    ]
    sf = StatsForecast(models=models, freq="D")

    h = 5
    n_paths = 10
    seed = 42

    sim_df = sf.simulate(h=h, df=df, n_paths=n_paths, seed=seed)

    # Check shape: 2 groups * 10 paths * 5 horizon = 100
    assert len(sim_df) == 100

    # Check columns
    expected_cols = {
        "unique_id",
        "ds",
        "sample_id",
        "AutoETS",
        "AutoARIMA",
        "CES",
        "AutoTheta",
    }
    assert expected_cols.issubset(set(sim_df.columns))

    # Check sample_id range
    assert sim_df["sample_id"].nunique() == n_paths
    assert sim_df["sample_id"].min() == 0
    assert sim_df["sample_id"].max() == n_paths - 1

    # Check unique_id
    assert set(sim_df["unique_id"]) == {1, 2}

    # Check reproducibility
    sim_df2 = sf.simulate(h=h, df=df, n_paths=n_paths, seed=seed)
    pd.testing.assert_frame_equal(sim_df, sim_df2)

    # Check different seed
    sim_df3 = sf.simulate(h=h, df=df, n_paths=n_paths, seed=seed + 1)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(sim_df, sim_df3)


def test_simulate_with_exog():
    # Create data with exogenous
    n_history = 100
    exog1 = np.random.randn(n_history)
    df = pd.DataFrame(
        {
            "unique_id": [1] * n_history,
            "ds": pd.to_datetime(["2020-01-01"] * n_history)
            + pd.to_timedelta(np.arange(n_history), unit="D"),
            "y": np.arange(n_history) + 10 * exog1 + 0.1 * np.random.randn(n_history),
            "exog1": exog1,
        }
    )

    h = 5
    X_df = pd.DataFrame(
        {
            "unique_id": [1] * h,
            "ds": pd.to_datetime(["2020-01-01"] * h)
            + pd.to_timedelta(np.arange(n_history, n_history + h), unit="D"),
            "exog1": np.random.randn(h),
        }
    )

    models = [AutoARIMA(season_length=1)]
    sf = StatsForecast(models=models, freq="D")

    n_paths = 3
    sim_df = sf.simulate(h=h, df=df, X_df=X_df, n_paths=n_paths, seed=42)

    assert len(sim_df) == h * n_paths
    assert "AutoARIMA" in sim_df.columns


def test_simulate_parallel():
    df = pd.DataFrame(
        {
            "unique_id": np.repeat(np.arange(10), 20),
            "ds": pd.to_datetime(["2020-01-01"] * 200)
            + pd.to_timedelta(np.tile(np.arange(20), 10), unit="D"),
            "y": np.random.randn(200),
        }
    )

    models = [AutoETS(season_length=1)]
    sf = StatsForecast(models=models, freq="D", n_jobs=2)

    h = 3
    n_paths = 2
    sim_df = sf.simulate(h=h, df=df, n_paths=n_paths, seed=42)

    # Check reproducibility with parallel
    sim_df2 = sf.simulate(h=h, df=df, n_paths=n_paths, seed=42)
    assert len(sim_df2) == 60
    pd.testing.assert_frame_equal(sim_df, sim_df2)


def test_simulate_edge_cases():
    # Need at least 15 observations for AutoETS to avoid 'tiny datasets' error
    df = pd.DataFrame(
        {
            "unique_id": [1] * 15,
            "ds": pd.to_datetime(["2020-01-01"] * 15)
            + pd.to_timedelta(np.arange(15), unit="D"),
            "y": np.arange(15),
        }
    )

    models = [AutoETS(season_length=1)]
    sf = StatsForecast(models=models, freq="D")

    # h=1, n_paths=1
    sim_df = sf.simulate(h=1, df=df, n_paths=1, seed=42)
    assert len(sim_df) == 1
    assert sim_df["sample_id"].iloc[0] == 0


def test_simulate_constant_series():
    df = pd.DataFrame(
        {
            "unique_id": [1] * 10,
            "ds": pd.to_datetime(["2020-01-01"] * 10)
            + pd.to_timedelta(np.arange(10), unit="D"),
            "y": [10.0] * 10,
        }
    )

    models = [AutoARIMA(season_length=1), AutoCES(season_length=1)]
    sf = StatsForecast(models=models, freq="D")

    sim_df = sf.simulate(h=3, df=df, n_paths=2, seed=42)
    assert len(sim_df) == 6
    np.testing.assert_allclose(sim_df["AutoARIMA"], 10.0)
    np.testing.assert_allclose(sim_df["CES"], 10.0)


def test_simulate_custom_aliases():
    df = pd.DataFrame(
        {
            "unique_id": [1] * 15,
            "ds": pd.to_datetime(["2020-01-01"] * 15)
            + pd.to_timedelta(np.arange(15), unit="D"),
            "y": np.arange(15),
        }
    )

    models = [AutoETS(season_length=1, alias="MyETS")]
    sf = StatsForecast(models=models, freq="D")

    sim_df = sf.simulate(h=2, df=df, n_paths=1)
    assert "MyETS" in sim_df.columns
    assert "AutoETS" not in sim_df.columns


def test_simulate_multiple_configs():
    df = pd.DataFrame(
        {
            "unique_id": [1] * 20,
            "ds": pd.to_datetime(["2020-01-01"] * 20)
            + pd.to_timedelta(np.arange(20), unit="D"),
            "y": np.arange(20) + np.random.normal(0, 0.1, 20),
        }
    )

    models = [
        AutoETS(season_length=1, alias="ETS_1"),
        AutoETS(season_length=1, alias="ETS_2"),
    ]
    sf = StatsForecast(models=models, freq="D")

    sim_df = sf.simulate(h=5, df=df, n_paths=2, seed=42)
    assert "ETS_1" in sim_df.columns
    assert "ETS_2" in sim_df.columns


if __name__ == "__main__":
    pytest.main([__file__])
