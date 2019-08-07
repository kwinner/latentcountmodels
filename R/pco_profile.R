if (!suppressWarnings(require('profvis', quietly = TRUE))) install.packages('profvis'); library('profvis')

source('pco-experiment.R')

params <- params.pco.default()
params$methods <- c('pco_rgd')

data   <- generate_data.pco(params)

gc()
# Rprof(tmp <- tempfile())
result <- trial.pco.fit(data, params)
# Rprof()
# p <- summaryRprof(tmp)
  
# htmlwidgets::saveWidget(p, "title")
# p