import warnings
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive
from statsforecast.utils import generate_series

def test_simulation_warning():
    df = generate_series(n_series=2, seed=42)
    models = [Naive()]
    sf = StatsForecast(models=models, freq='D', n_jobs=1)
    
    # Total points = n_series (2) * n_paths (1000) * h (51) = 102,000 > 100,000
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sf.simulate(h=51, df=df, n_paths=1000)
        
        # Check if the warning was issued
        assert len(w) >= 1
        assert "Large simulations may consume significant memory and time" in str(w[-1].message)

if __name__ == "__main__":
    test_simulation_warning()
    print("Test passed!")
