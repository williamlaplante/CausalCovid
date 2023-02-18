library(EpiEstim)
data_source <- "https://raw.githubusercontent.com/ccodwg/CovidTimelineCanada/main/data/can/cases_can.csv"
data <- read.csv(data_source)

res <- estimate_R(data$value_daily, method="parametric_si", config=make_config(list(mean_si = 4.0, std_si = 8.0)))

write.csv(res$R, "results_R.csv")