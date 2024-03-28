
# TBATS R TEST ---- 
## Generates a TBATS forecast for a given dataset
## Assumes the train datasets already exist 

library(data.table)
library(forecast)
library(parallel)
library(tidyverse)

fit_predict <- function(group, unique_id, ds, y, horizon, frequency) {
  tryCatch({
    if(group %in% c("Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Other")){
      y_ts <- ts(as.numeric(y), frequency = frequency)  
    } else {
      # "Hourly"
      y_ts <- msts(as.numeric(y), seasonal.periods=frequency) # frequency here is a vector 
    }
    model <- tbats(y_ts)
    fcst <- forecast(model, horizon)$mean  
    return(data.table(unique_id = unique_id, y = fcst, id = seq_along(fcst)))
  }, error = function(e) {
    return(data.table(unique_id = unique_id, y = NA, id = 1:horizon)) 
  })
}

generate_tbats <- function(dataset, group, horizon, frequency){
  df <- fread(paste0("data/", dataset, "-", group, "-train.csv"))
  series <- df[, .(ds = list(ds), y = list(y)), by = unique_id]
  
  start <- Sys.time()
  fcst <- mclapply(seq_along(series$unique_id), function(i) {
    fit_predict(group, series$unique_id[i], series$ds[[i]], series$y[[i]], horizon, frequency)
  })
  time_taken <- difftime(Sys.time(), start, units = "secs")
  
  forecast <- rbindlist(lapply(fcst, function(df) {
    # tbats returns NA for some series 
    if(is.ts(df$y)) {
      df$y <- as.numeric(df$y)
    }
    return(df)
  }))
  
  names(forecast) <- c("unique_id", "R-TBATS", "id")
  rownames(forecast) <- NULL
  fwrite(forecast, paste0("data/R-forecasts-", dataset, "-", group, ".csv"))
  
  time_df <- data.frame(time = time_taken, model = "R")
  fwrite(time_df, paste0("data/R-time-", dataset, "-", group, ".csv"))
}

# M3 ---- 
generate_tbats("M3", "Yearly", 6, 1)
generate_tbats("M3", "Quarterly", 8, 4)
generate_tbats("M3", "Monthly", 18, 12)
generate_tbats("M3", "Other", 8, 1)

# M4 ---- 
generate_tbats("M4", "Yearly", 6, 1)
generate_tbats("M4", "Quarterly", 8, 4)
generate_tbats("M4", "Monthly", 18, 12)
generate_tbats("M4", "Weekly", 13, 1)
generate_tbats("M4", "Daily", 14, 7)
generate_tbats("M4", "Hourly", 48, c(24, 24*7))
