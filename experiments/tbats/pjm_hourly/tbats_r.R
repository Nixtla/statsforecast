library(dplyr)
library(forecast)
library(readr)

df <- read_csv(
  "https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/PJM_Load_hourly.csv",
  col_names = c("ds", "y"),
  col_types = "cn", 
  skip = 1, # skip original header
) %>%
  arrange(ds) %>%
  mutate(unique_id = "PJM_Load_hourly")

n_windows <- 5 
time <- 0
res <- list()

for(i in 1:n_windows){
  y_train <- df %>%
    head(-24*i) %>%
    pull(y) %>%
    msts(seasonal.periods=c(24, 24 * 7))
  
  start <- Sys.time() 
  forecasts <- tbats(
    y_train
  ) %>%
    forecast(24)
  end <- Sys.time()
  
  time <- time+difftime(end, start, units="mins")

  res[[n_windows+1-i]] <- forecasts$mean 
}

fcst_df <- df %>%
  tail(24*n_windows) %>%
  mutate(tbats_r = c(t(do.call(rbind, res))))

fcst_df %>% 
  write_csv("data/tbats-r.csv")

tibble(time=time, model='tbats_r') %>%
  write_csv("data/tbats-r-time.csv")
