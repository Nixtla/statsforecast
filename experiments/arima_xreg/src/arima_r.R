library(dplyr)
library(readr)
library(forecast)
library(tidyr)
library(stringr)

args <- commandArgs(trailingOnly=TRUE)

df <- read_csv(str_glue('data/EPF-{args[1]}.csv'))
x <- df %>%
	pull(y) %>%
	ts(frequency=24)
xreg <- data.matrix(df[-3:-1])
future_xreg <- read_csv(str_glue('data/EPF-{args[1]}-test.csv'))
future_xreg <- data.matrix(future_xreg[-3:-1])

start <- Sys.time() 
forecasts <- x %>%
	auto.arima(xreg=xreg, trace=T, allowdrift=FALSE, allowmean=FALSE,
		   approximation=FALSE) %>%
	forecast(24*7, xreg=future_xreg) %>%
	.$mean
end <- Sys.time()

forecasts %>% 
  write.table(str_glue('data/arima-r-forecasts-{args[1]}.txt'), 
	      row.name=F, col.name=F)

tibble(time=difftime(end, start, units="secs"), model='auto_arima_r') %>%
  write_csv(str_glue('data/arima-r-time-{args[1]}.csv'))
