library(dplyr)
library(readr)
library(future)
library(tsibble)
library(lubridate)
library(fable)
library(tidyr)
library(purrr)
library(furrr)
library(stringr)

args <- commandArgs(trailingOnly=TRUE)

meta <- list(
  Weekly=list(horizon=13, func_ds=yearweek),
  Hourly=list(horizon=48, func_ds=function(x) x), # The official docs tell to convert to posix however the read data are just integers
  Daily=list(horizon=14, func_ds=as_date),
  Yearly=list(horizon=6, func_ds=year),
  Monthly=list(horizon=18, func_ds=yearmonth),
  Quarterly=list(horizon=8, func_ds=yearquarter)
)
horizon <- meta[[args[4]]][['horizon']]
func_ds <- meta[[args[4]]][['func_ds']]

df <- read_csv(str_glue('data/{args[2]}-{args[4]}.csv'))
# plan(multisession, gc=TRUE)
# This actually makes an slower perfomance from fable
# This is an old issue but I don't know if still this applies
# https://github.com/tidyverts/fable/issues/169

start <- Sys.time() 
forecasts <- df |>
    mutate(ds = func_ds(ds)) |>
    as_tsibble(key = unique_id, index = ds) |>
    model(arima = ARIMA(y)) |>
    forecast(h = horizon) |>
    as_tibble() |>
    select(unique_id, ds, arima_fable =.mean)
end <- Sys.time()

forecasts %>% 
  write.table(
    str_glue('data/fable-arima-r-forecasts-{args[2]}-{args[4]}.txt'),
    row.name=F, col.name=F
  )

tibble(time=difftime(end, start, units="secs"), model='auto_arima_fable_r') %>%
  write_csv(str_glue('data/fable-arima-r-time-{args[2]}-{args[4]}.csv'))
