import warnings

from nbdev.showdoc import show_doc
warnings.simplefilter(action='ignore', category=FutureWarning)

show_doc(AutoARIMAProphet, title_level=3)
show_doc(AutoARIMAProphet.fit, title_level=3)
show_doc(AutoARIMAProphet.predict, title_level=3)
df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
df.head()
m = Prophet(daily_seasonality=False)
m.fit(df)
future = m.make_future_dataframe(365)
forecast = m.predict(future)
fig = m.plot(forecast)
fig = m.plot(forecast)
playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))
fig = m.plot(forecast)
fig = m.plot(forecast)

