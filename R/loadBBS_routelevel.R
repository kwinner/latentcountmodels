library(reshape2)
library(unmarked)
source('pcountOpen_constrained_apgffwd.R')
# library(tidyr)

file.name <- "~/Work/Data/BBS/massach.csv"

# species.names <- c("American Robin",
#                    "Common Grackle",
#                    "European Starling",
#                    "House Sparrow",
#                    "Red-winged Blackbird",
#                    "Wood Thrush")
species.names <- c("House Sparrow",
                   "Ovenbird",
                   "Wood Thrush")
# species.AOUs <- c(07610, # American Robin
#                   05110, # Common Grackle
#                   04930, # European Starling
#                   06882, # House Sparrow
#                   04980, # Red-Winged Blackbird
#                   07550) # Wood Thrush
species.AOUs <- c(06882, # House Sparrow
                  06740, # Ovenbird
                  07550) # Wood Thrush

years <- seq(2013, 2018)

stopifnot(length(species.names) == length(species.AOUs))
n.species <- length(species.names)
n.years <- length(years)

data <- read.csv(file.name)

data.pruned <- data[which((data$AOU %in% species.AOUs) & (data$Year %in% years)),]

valid.RPIDs <- c(101) # stick to regular observation protocols
data.pruned <- data.pruned[which(data.pruned$RPID %in% valid.RPIDs),]

# models <- vector(n.species, mode='list')
# rt     <- vector(n.species, mode='numeric')
for(i.species in 1:n.species) {
# for(i.species in 6) {
  species.name <- species.names[i.species]
  species.AOU  <- species.AOUs[i.species]
  
  print(paste0('Fitting model for ', species.name, '...'))
  
  data.species <- data.pruned[which(data.pruned$AOU == species.AOU),]
  
  # melt.species <- melt(data.species, id=c("Route", "Year"))
  # acast(melt.species, Route~Year, value.var = "SpeciesTotal")
  
  # melt.species <- gather(data.species, Route, Year, SpeciesTotal)
  # ym <- spread(melt.species, Route, Year)
  
  species.y <- acast(data.species, Route~Year, value.var = "SpeciesTotal")
  
  # valid.routes <- which(apply(species.y, 1, function(x) sum(!is.na(x))) == length(years))
  valid.routes <- which(apply(species.y, 1, function(x) sum(!is.na(x)) / length(years) > 0.5))
  species.y.valid <- species.y[valid.routes,]
  
  # print(length(valid.routes))
  # print(valid.routes)
  
  print(sum(species.y, na.rm = TRUE))
  
  # browser()
  
  # fix.rho.val <- 0.2
  # dynamics    <- "notrend"
  # mixture     <- "P"
  # immigration <- FALSE
  # apgffwd.method <- "L-BFGS-B"
  # 
  # umf <- unmarkedFramePCO(y          = species.y.valid,
  #                         numPrimary = ncol(species.y.valid))
  # 
  # time.start <- proc.time()[3]
  # models[[i.species]] <- pcountOpen_apgffwd(~1, ~1, ~1, ~1, umf, K=K, immigration = immigration, se= FALSE,
  #                                           dynamics = dynamics, mixture = mixture,
  #                                           method = apgffwd.method,
  #                                           # fix.rho = fix.rho.val)
  #                                           fix.lambda = 'auto')
  # rt[i.species] <- proc.time()[3] - time.start
}

# browser()