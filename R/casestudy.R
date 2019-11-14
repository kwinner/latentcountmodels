library(reshape2)
library(unmarked)
source('pcountOpen_constrained_apgffwd.R')

clear_data = FALSE # whether to clear results
skip_trial = TRUE  # whether to skip trials already found in results (irrelevant if clear_data is TRUE)

# start.years <- seq(2013, 1965, -12)
# start.years <- seq(1965, 2013, 12)
start.years <- c(1965)
end.year <- 2018
n.start.years <- length(start.years)

NA.percentage <- 0.5

all.years <- seq(min(start.years), end.year)

species.names <- c(
  #"House Sparrow",
  "Ovenbird",
  "Wood Thrush",
NULL)
species.AOUs <- c(
  #06882, # House Sparrow
  06740, # Ovenbird
  07550, # Wood Thrush
NULL)
min_rho_assumption <- c("House Sparrow" = 0.5,
                        "Ovenbird"      = 0.3,
                        "Wood Thrush"   = 0.1)

model.descriptions <- data.frame(matrix(ncol=3, nrow=0))
colnames(model.descriptions) <- c("dynamics", "immigration", "mixture")
model.descriptions[nrow(model.descriptions)+1,] <- c("notrend",  FALSE, 'P')
model.descriptions[nrow(model.descriptions)+1,] <- c("constant", FALSE, 'P')
model.descriptions[nrow(model.descriptions)+1,] <- c("trend",    FALSE, 'P')
model.descriptions[nrow(model.descriptions)+1,] <- c("autoreg",  FALSE, 'P')
model.descriptions[nrow(model.descriptions)+1,] <- c("trend",    TRUE,  'P')
model.descriptions[nrow(model.descriptions)+1,] <- c("autoreg",  TRUE,  'P')
n.models <- nrow(model.descriptions)

methods <- c(
  "apgffwd",
  # "trunc",
NULL)
n.methods <- length(methods)
apgffwd.optimizer <- 'L-BFGS-B'
trunc.optimizer   <- 'Nelder-Mead'

stopifnot(length(species.names) == length(species.AOUs))
n.species <- length(species.names)

# read in all the data, prune to species, years of interest
file.name <- "~/Work/Data/BBS/massach.csv"

data <- read.csv(file.name)

data.pruned <- data[which((data$AOU %in% species.AOUs) & (data$Year %in% all.years)),]

valid.RPIDs <- c(101) # stick to regular observation protocols
data.pruned <- data.pruned[which(data.pruned$RPID %in% valid.RPIDs),]

# remove data whose route or year are NA (would probably get removed incidentally later, but causes trouble with acast)
data.pruned <- data.pruned[which(!is.na(data.pruned$Route) & !is.na(data.pruned$Year)),]

result.names <- c("species", "start.year", "method", "dynamics", "imm", "mixture", "R", "iota", "rho", "AIC", "K", "rt", "model.index")
if(clear_data) {
  confirm = readline(prompt="Really clear results? (y/Y/yes to confirm)")
  if(trimws(confirm) %in% c("y", "Y", "yes")) {
    models <- list() # raw list of models for debugging
    results <- data.frame(matrix(ncol = length(result.names), nrow = 0)) # structured output data frame
    colnames(results) <- result.names
  }
}

for(i.species in 1:n.species) {
  species.name <- species.names[i.species]
  species.AOU  <- species.AOUs[i.species]

  data.species <- data.pruned[which(data.pruned$AOU == species.AOU),]

  for(i.start_year in 1:n.start.years) {
    start.year <- start.years[i.start_year]
    years <- seq(start.year, end.year)

    # subset the data, convert to a matrix format
    data.species.subset <- data.species[which(data.species$Year %in% years),]
    species.y <- acast(data.species.subset, Route~Year, value.var = "SpeciesTotal")

    # remove routes which have too many NAs or which begin with NAs
    # valid.routes <- which(apply(species.y, 1, function(x) (sum(!is.na(x)) / length(years) > NA.percentage) & !is.na(x[1])))
    valid.routes <- which(apply(species.y, 1, function(x) (sum(!is.na(x)) / length(years) > NA.percentage)))
    species.y.valid <- species.y[valid.routes,]

    umf <- unmarkedFramePCO(y          = species.y.valid,
                            numPrimary = ncol(species.y.valid))
    
    print(paste0("For ", species.name, ", ", start.year, "-", end.year, " there are ", length(valid.routes), " routes and Y=", sum(species.y.valid, na.rm = TRUE)))
    
    for(i.method in 1:n.methods) {
      method <- methods[i.method]
      
      for(i.model in 1:n.models) {
        dynamics <- model.descriptions[i.model, 'dynamics']
        imm      <- as.logical(model.descriptions[i.model, 'immigration'])
        mixture  <- model.descriptions[i.model, 'mixture']
        
        dynamics.print <- dynamics
        if(imm)
          dynamics.print <- paste0(dynamics.print, '+imm')
        
        tryCatch({
          if(identical(method, 'apgffwd')) {
            # check whether this result already exists
            if(skip_trial && any(results$species == species.name &
                                 results$start.year == start.year &
                                 results$method == method &
                                 results$dynamics == dynamics &
                                 results$imm == imm &
                                 results$mixture == mixture)) {
              print(paste0("Result for ", dynamics.print, " with apgffwd found, skipping trial."))
              next()
            }
            
            print(paste0("Fitting model for ", dynamics.print, " with apgffwd."))
            
            time.start <- proc.time()[3]
            m <- pcountOpen_apgffwd(~1, ~1, ~1, ~1, umf, se= FALSE,
                                    dynamics = dynamics, immigration = imm, mixture = mixture,
                                    nll.fun = 'apgffwd', method = apgffwd.optimizer,
                                    fix.lambda = 'auto')
            rt.iter <- proc.time()[3] - time.start
            
            K <- NA # just for results saving purposes
          } else if(identical(method, 'trunc')) {
            K <- K.default(species.y.valid, p = min_rho_assumption[species.name])
            
            if(skip_trial && any(results$species == species.name &
                                 results$start.year == start.year &
                                 results$method == method &
                                 results$dynamics == dynamics &
                                 results$imm == imm &
                                 results$mixture == mixture &
                                 results$K == K)) {
              print(paste0("Result for ", dynamics.print, " with trunc, K=", K, " found, skipping trial."))
              next()
            }
            
            print(paste0("Fitting model for ", dynamics.print, " with trunc, K=", K))
            
            time.start <- proc.time()[3]
            m <- pcountOpen_apgffwd(~1, ~1, ~1, ~1, umf, se= FALSE,
                                    dynamics = dynamics, immigration = imm, mixture = mixture,
                                    nll.fun = 'trunc', method = trunc.optimizer,
                                    maxit = 1500, K = K,
                                    fix.lambda = 'auto')
            rt.iter <- proc.time()[3] - time.start
          }
          models[[length(models)+1]] <- c(species=species.name, start.year=start.year, method=method, dynamics=dynamics, imm=imm, mixture=mixture, model=m)
          if(m@opt$convergence == 0)
            print("Model fit successfully.")
          else
            print("Model did not converge.")
          
          AIC.iter <- m@AIC
          
          # Get (or derive) model parameters R, iota, rho
          if(identical(method, 'apgffwd')) {
            if(identical(dynamics, "constant")) {
              R    <- coef(m, 'omega')
              rho  <- coef(m, 'det')
              iota <- coef(m, 'gamma')
            } else if(identical(dynamics, "notrend")) {
              R    <- coef(m, 'omega')
              rho  <- coef(m, 'det')
              iota <- (1 - R) / rho
            } else if(identical(dynamics, "trend")) {
              R    <- coef(m, 'gamma')
              rho  <- coef(m, 'det')
              iota <- coef(m, 'iota')
            } else if(identical(dynamics, "autoreg")) {
              R    <- coef(m, 'omega') + coef(m, 'gamma')
              rho  <- coef(m, 'det')
              iota <- coef(m, 'iota')
            }
          } else if(identical(method, 'trunc')) {
            if(identical(dynamics, "constant")) {
              R    <- plogis(coef(m, 'omega'))
              rho  <- plogis(coef(m, 'det'))
              iota <- exp(coef(m, 'gamma'))
            } else if(identical(dynamics, "notrend")) {
              R    <- plogis(coef(m, 'omega'))
              rho  <- plogis(coef(m, 'det'))
              iota <- (1 - R) / rho
            } else if(identical(dynamics, "trend")) {
              R    <- exp(coef(m, 'gamma'))
              rho  <- plogis(coef(m, 'det'))
              if(is.null(coef(m, 'iota')))
                iota <- NA
              else
                iota <- exp(coef(m, 'iota'))
            } else if(identical(dynamics, "autoreg")) {
              R    <- plogis(coef(m, 'omega')) + exp(coef(m, 'gamma'))
              rho  <- plogis(coef(m, 'det'))
              if(is.null(coef(m, 'iota')))
                iota <- NA
              else
                iota <- exp(coef(m, 'iota'))
            }
          }
          
          if(is.null(iota))
            iota <- NA
          
          result <- c(species=species.name, start.year=start.year, method=method, 
                      dynamics=dynamics, imm=imm, mixture=mixture,
                      R=R, iota=iota, rho=rho, AIC=AIC.iter, K = K, rt=rt.iter,
                      model.index = length(models))
          names(result) <- result.names
          
          results[nrow(results)+1,] <- result
        }, error = function(e) {
          print(paste0("Something went wrong: ", e))
        })
      }
    }
    browser()
  }
  browser()
}