species.names <- c(
  "House Sparrow",
  "Ovenbird",
  "Wood Thrush",
NULL)
methods <- c(
  # "apgffwd",
  "trunc",
NULL)
# start.years <- c(1965, 2013)
start.years <- 2013

# species <- "House Sparrow"
# method  <- "apgffwd"
# start.year <- 1965

# min.AIC.results <- data.frame()
for(species in species.names) {
  for(method in methods) {
    for(start.year in start.years) {
      subset.results <- results[which(results$species == species & results$method == method & results$start.year == start.year),]
      
      min.AIC <- subset.results[which(subset.results$AIC == min(subset.results$AIC)),c('species', 'start.year', 'method', 'dynamics', 'imm', 'R', 'iota', 'rho', 'AIC')]
      total.rt <- sum(as.numeric(subset.results$rt))
      avg.rt <- total.rt / nrow(subset.results)
      
      min.AIC$avg.rt <- avg.rt
      
      min.AIC.results <- rbind(min.AIC.results, min.AIC)
    }
  }
}