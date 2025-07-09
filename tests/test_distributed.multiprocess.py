%load_ext autoreload
%autoreload 2
from fastcore.test import test_eq
from nbdev.showdoc import add_docs, show_doc
show_doc(MultiprocessBackend, title_level=3)
from statsforecast import StatsForecast
from statsforecast.models import Naive
from statsforecast.utils import generate_series
df = generate_series(10).reset_index()
df['unique_id'] = df['unique_id'].astype(str)

class FailNaive:
    def forecast(self):
        pass
    def __repr__(self):
        return 'Naive'

def test_mp_back(n_jobs=1):
    backend = MultiprocessBackend(n_jobs=n_jobs)
    #forecast
    fcst = backend.forecast(df, models=[Naive()], freq='D', h=12)
    fcst_stats = StatsForecast(models=[Naive()], freq='D').forecast(df=df, h=12)
    test_eq(fcst, fcst_stats)
    #crossvalidation
    fcst = backend.cross_validation(df, models=[Naive()], freq='D', h=12)
    fcst_stats = StatsForecast(models=[Naive()], freq='D').cross_validation(df=df, h=12)
    test_eq(fcst, fcst_stats)
    # fallback model
    fcst = backend.forecast(df, models=[FailNaive()], freq='D', fallback_model=Naive(), h=12)
    fcst_stats = StatsForecast(models=[Naive()], freq='D').forecast(df=df, h=12)
    test_eq(fcst, fcst_stats)
    
    #cross validation
    fcst_fugue = backend.cross_validation(df, models=[FailNaive()], freq='D', fallback_model=Naive(), h=12)
    fcst_stats = StatsForecast(models=[Naive()], freq='D').cross_validation(df=df, h=12)
    test_eq(fcst_fugue, fcst_stats)

test_mp_back()
#| eval: false
test_mp_back(n_jobs=10)
