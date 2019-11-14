species.names <- c("American Robin", "Common Grackle", "European Starling", "House Sparrow", "Red-winged Blackbird", "Wood Thrush")
states <- c("MA", "CT", "RI", "NH", "VT", "ME")
years <- seq(2000, 2010)

rho.values <- seq(0.1, 0.9, 0.1)

bbs.dir <- "~/Work/Data/BBS/"
file.suffix <- "_counts.bbs"

# data <- read.table(paste0, sep='\t', header = TRUE)

# colnames(test) <- sapply(colnames(test), function(x) gsub('X', '', x))

# stateA <- states[1]
# stateB <- states[2]
# 
# dataA <- read.table(paste0(bbs.dir, stateA, file.suffix), sep='\t', header = TRUE)
# dataB <- read.table(paste0(bbs.dir, stateB, file.suffix), sep='\t', header = TRUE)
# 
# colnames(dataA) <- sapply(colnames(dataA), function(x) gsub('X', '', x))
# colnames(dataB) <- sapply(colnames(dataB), function(x) gsub('X', '', x))
# 
# dataC <- cbind(list(state="MA"), dataA)

data.all <- NULL
for(state in states) {
  data <-read.table(paste0(bbs.dir, state, file.suffix), 
                    sep='\t', header = TRUE, strip.white = TRUE)
  colnames(data) <- sapply(colnames(data), 
                           function(x) gsub('X', '', x))
  data <- cbind(list(state=state), data)
  
  data.all <- rbind(data.all, data)
}

data.pruned <- NULL
for(species in species.names) {
  data.pruned <- rbind(data.pruned, 
                       data.all[which(as.character(data.all$species) == species),
                                c("state", "species", as.character(years))])
}

route.counts <- data.all[which(as.character(data.all$species) == "Route Count"),
                         c("state", as.character(years))]

parameter.estimates <- setNames(data.frame(matrix(ncol = 3, nrow = 0)), c("species", "param", "estimate"))

# data.normalized <- round(data.pruned[,as.character(years)] / do.call("rbind", replicate(length(species.names), route.counts, simplify=FALSE))[,as.character(years)])
for(species in species.names) {
  data.species <- data.pruned[which(data.pruned$species == species), c("state", as.character(years))]
  
  if(species == 'Wood Thrush')
    valid.states <- which(data.species$state != 'RI')
  else
    valid.states <- 1:nrow(data.species)
  
  data.normalized <- round(data.species[valid.states,as.character(years)] / route.counts[valid.states,as.character(years)])
  
  umf <- unmarkedFramePCO(y          = data.normalized,
                          numPrimary = ncol(data.normalized))
  
  print(data.normalized)
}


# library(unmarked)
# source('pcountOpen_constrained_apgffwd.R')
# dynamics    <- "autoreg"
# immigration <- TRUE
# apgffwd.method <- "L-BFGS-B"
# 
# fix.rho.val <- 0.5
# 
# umf <- unmarkedFramePCO(y          = data.normalized,
#                         numPrimary = ncol(data.normalized))
# 
# time.start <- proc.time()[3]
# m <- pcountOpen_apgffwd(~1, ~1, ~1, ~1, umf, K=K, immigration = immigration, dynamics = dynamics, se= FALSE,
#                         method = apgffwd.method,
#                         fix.rho = fix.rho.val)
# rt <- proc.time()[3] - time.start