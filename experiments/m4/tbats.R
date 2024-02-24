library(data.table)
library(forecast)
library(parallel)

dt <- fread("https://datasets-nixtla.s3.amazonaws.com/m4-hourly.csv")
horizon <- 24L
dt[, cutoff := .N - horizon, by = unique_id]
train <- dt[ds <= cutoff]
valid <- dt[ds > cutoff]
series <- train[, .(ys = list(y)), by = unique_id][, ys]

fit_predict <- function(y, horizon) {
  y <- msts(y, seasonal.periods = c(24, 24 * 7))
  model <- tbats(y)
  forecast(model, horizon)$mean
}

start <- Sys.time()
preds <- mclapply(series, fit_predict, horizon = horizon)
time_taken <- difftime(Sys.time(), start, units = "mins")

valid[, y_pred := unlist(preds)]
valid[, smape := 2 * abs(y - y_pred) / (abs(y) + abs(y_pred))]
avg_smape <- valid[, .(avg_smape = mean(smape)), by = unique_id][, mean(avg_smape)]
print(time_taken)
cat("Average SMAPE:", round(100 * avg_smape, 1))
