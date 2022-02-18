library(dplyr)
library(readr)
library(future)
library(forecast)
library(tidyr)
library(purrr)
library(furrr)
library(stringr)

args <- commandArgs(trailingOnly=TRUE)
meta <- list(
  Weekly=list(horizon=13, seasonality=1),
  Hourly=list(horizon=48, seasonality=24),
  Daily=list(horizon=14, seasonality=1),
  Yearly=list(horizon=6, seasonality=1),
  Monthly=list(horizon=18, seasonality=12),
  Quarterly=list(horizon=8, seasonality=4)
)
horizon <- meta[[args[1]]][['horizon']]
seasonality <- meta[[args[1]]][['seasonality']]

df <- read_csv(str_glue('data/M4-{args[1]}.csv'))

plan(multiprocess, gc = TRUE)

split_tibble <- function(tibble, col = 'col') tibble %>% split(., .[, col])
dflist <- split_tibble(df, 'unique_id')

start <- Sys.time() 
forecasts <- dflist %>%
  future_map(~{
    .x %>%
      pull(y) %>%
      ts(frequency=seasonality) %>%
      auto.arima(allowdrift=FALSE, allowmean=FALSE, approximation=FALSE) %>%
      forecast(horizon) %>%
      .$mean
  }) %>%
  reduce(rbind)
end <- Sys.time()

forecasts %>% 
  write.table(str_glue('data/arima-r-forecasts-M4-{args[1]}.txt'), 
	      row.name=F, col.name=F)

tibble(time=difftime(end, start, units="secs"), model='auto_arima_r') %>%
  write_csv(str_glue('data/arima-r-time-M4-{args[1]}.csv'))
