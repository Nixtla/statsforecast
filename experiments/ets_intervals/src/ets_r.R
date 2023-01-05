library(dplyr)
library(readr)
library(future)
library(forecast)
library(tidyr)
library(purrr)
library(furrr)
library(stringr)
library(data.table)

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

conf_levels <- seq(55,95,5)

start <- Sys.time() 
forecasts <- dflist %>%
  future_map(~{
    .x %>%
      pull(y) %>%
      ts(frequency=seasonality) %>%
      ets() %>%
      forecast(horizon, level = c(conf_levels)) %>% 
      .[c("mean", "lower", "upper")] %>%
      reduce(cbind) 
  }) %>% 
  reduce(rbind)
end <- Sys.time()

forecasts <- as.data.frame(forecasts)
forecasts <- forecasts %>% mutate(unique_id = rep(names(dflist), each=horizon))

forecasts <- forecasts %>% select(c(unique_id, everything()))
colnames(forecasts) <- c("unique_id", "ets-r_mean", paste0("ets-r_lowerb_", conf_levels), paste0("ets-r_upperb_", conf_levels))

forecasts %>% 
  write.table(str_glue('data/ets-r-forecasts-M4-{args[1]}-pred-int.csv'), 
	      row.name=F, col.name=T, quote=F, sep=",")

tibble(time=difftime(end, start, units="secs"), model='ets_r') %>%
  write_csv(str_glue('data/ets-r-time-M4-{args[1]}-pred-int.csv'))

