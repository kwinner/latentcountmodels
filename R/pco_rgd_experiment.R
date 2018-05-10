# PCO vs PCO_RGD comparison

rm(list=ls())

library(unmarked)
source('pcountOpen_rgdual.R')

params.default <- function() {
  params <- list()
  
  params$methods <- c('pco_trunc:default', 'pco_trunc:ratio', 'pco_rgd') # which methods to test
  params$response.vars <- c('nll', 'rt', 'n.iters', 'fit')        # which values to measure
  
  params$M <- 2 # num of sites
  params$T <- 5 # num of timesteps
  
  params$n.samples <- 1  # num of samples to generate
  params$n.iters   <- 2  # num of times to replicate each trial
  
  params$shared.data <- 'trial' # {'trial', 'experiment', or 'none'}
                                # if 'trial', then data is shared within each trial
                                # if 'experiment', then data is shared across all trials for the same experiment value
                                # if 'none', then new data is generated for each trial
  params$experiment.variable <- 'rho.gen' # the variable to be varied experimentally
  params$experiment.values   <- seq(0.05, 0.95, by=0.45)
  
  params$gamma.gen <- 0.5 # mean offspring
  params$omega.gen <- 0.5 # survival probability
  params$rho.gen   <- 0.8 # detection probability
  params$iota.gen  <- 5  # immigration rate
  
  params$dynamics    <- 'trend'
  params$immigration <- TRUE
  
  params$arrival.gen   <- 'poisson'   # arrival distribution
  params$offspring.gen <- 'poisson'   # offspring distribution
  params$survival.gen  <- 'bernoulli' # offspring distribution
  
  return(params)
}

# meta parameters are ones which (presumably) have no bearing on the actual experiment
# and do not need to be recorded, including filesystem defaults
params.meta.default <- function() {
  params.meta <- list()
  
  params.meta$silent <- FALSE
  params.meta$write.results <- TRUE
  
  params.meta$data.dir           <- ''
  params.meta$results.parent.dir <- '/Users/kwinner/Work/Data/Results'
  params.meta$experiment.name    <- "NULL"
  
  params.meta$timestamp.format <- "%Y-%m-%d %H:%M:%OS6"
  
  return(params.meta)
}

# parameters for plotting and summarizing data
params.analysis.default <- function() {
  params.analysis <- list()
  
  return(params.analysis)
}

run_experiment <- function(experiment.fun = pco_rgd_experiment, 
                           params         = params.default(), 
                           params.meta    = params.meta.default()) {
  # check that the parent result directory exists (and create it if it doesn't)
  if (params.meta$write.results) {
    if (!file.exists(params.meta$results.parent.dir)) {
      tryCatch({dir.create(params.meta$results.parent.dir)},
      error = function(e) {
        warning('Unable to find/create result directory. Results will not be saved.')
        params.meta$write.results <- FALSE
      })
    }
  }
  
  # create a directory for this experiment
  if (params.meta$write.results) {
    timestamp <- strftime(Sys.time(), params.meta$timestamp.format)
    if (is.null(params.meta$experiment.name))
      params.meta$results.dir <- timestamp
    else
      params.meta$results.dir <- paste(timestamp, ' "', params.meta$experiment.name ,'"', sep="")
    params.meta$results.path <- file.path(params.meta$results.parent.dir, params.meta$results.dir)
    
    if (!file.exists(params.meta$results.path)) {
      tryCatch({dir.create(params.meta$results.path)},
      error = function(e) {
        warning('Unable to find/create result directory. Results will not be saved.')
        params.meta$write.results <- FALSE
      })
    } else
      warning('Warning: result directory already exists, results may be overwritten.')
  }
  
  result <- experiment.fun()
  
  if (params.meta$write.results) {
    write.table(result, file = file.path(params.meta$results.path, 'results.df'))
    
    write.params(params,      'params.txt')
    write.params(params.meta, 'params.meta.txt')
  }
  
  return(list(result=result,
              params.meta = params.meta))
}

pco_rgd_experiment <- function(params      = params.default(), 
                               params.meta = params.meta.default()) {
  result <- data.frame()
  
  # to prevent confusion, "clear" the setting of experiment.variable in the experiment params
  params[[params$experiment.variable]] <- NA
  
  # generate data (to be shared across all values of experiment.variable)
  if (params$shared.data == 'experiment') {
    # NOTE: experiment-wide data sharing should only be used if the experiment variable does not affect data generation
    if (any(grepl(c('gen', 'M', 'T'), params$experiment.variable)))
      warning(sprintf("Experiment-wide data sharing is generally inappropriate when the experiment variable (%s) affects data generation.", params$experiment.variable))
    data <- generate_data(params)
  }
  
  # for approximating remaining runtime
  start.time <- proc.time()[3]
  
  for (i.experiment.value in 1:length(params$experiment.values)) {
    experiment.value <- params$experiment.values[[i.experiment.value]]
    
    # create a new params object w/ the experiment variable
    params.trial <- params
    params.trial[[params$experiment.variable]] <- experiment.value
    
    # generate data (to be shared across all trials)
    if (params$shared.data == 'trial')
      data <- generate_data(params.trial)
    
    for (i.trial in 1:params$n.iters) {
      if (!params.meta$silent)
        cat(sprintf("Trial #%d, %s = %s...", i.trial, params$experiment.variable, toString(experiment.value)))
      
      # generate trial-specific data (shared only across the different methods)
      if (params$shared.data == 'none')
        data <- generate_data(params.trial)
      
      tryCatch({
        # run the trial
        result.trial <- pco_rgd_trial(params.trial, data)
        
        # add the experiment value and reorder the columns
        result.trial[,params$experiment.variable] <- experiment.value
        result.trial <- result.trial[,c(ncol(result.trial), 1:(ncol(result.trial) - 1))]
        
        # append this trial to the record
        result <- rbind(result, result.trial)
        
        # print the runtime, approximate remaining time
        overall.elapsed.time <- proc.time()[3] - start.time
        percent.completed <- ((((i.experiment.value - 1) * params$n.iters) + i.trial) / 
                              (length(params$experiment.values) * params$n.iters))
        expected.remaining.time <- overall.elapsed.time * (1 - percent.completed) / percent.completed
        
        if (!params.meta$silent)
          cat(sprintf(' remaining time ~= %s\n', s2hms(expected.remaining.time)))
      }, error = function(e) {
        if (!params.meta$silent)
          cat(sprintf(' Error: %s.\n', e))
      })
    }
  }
  
  return(result)
}

pco_rgd_trial <- function(params = params.default(), 
                          data   = NULL) {
  # generate data
  if(is.null(data)) {
    stopifnot(is.null(params$n.samples) || params$n.samples == 1)
    params$n.samples <- 1
    data <- generate_data(params)
    data.sampled = TRUE
  } else
    data.sampled = FALSE
  
  # assert that the parameters didn't change
  stopifnot(all(mapply(function(a, b) {all(a == b)}, params, data$params)))
  
  # build an unmarkedframe
  umf <- unmarkedFramePCO(y = data$y, numPrimary = params$T)

  # prepare results
  result <- data.frame()
  for (i.method in 1:length(params$methods)) {
    method <- params$methods[[i.method]]
    
    result[i.method, 'method'] <- method
    
    # use the truncated (original) method
    if (grepl('pco_trunc', method)) {
      # set K
      if (grepl('default', method)) {
        K <- max(data$y) + 20
      } else if (grepl('ratio', method)) {
        ratio <- 0.38563049853372433
        result[i.method, 'ratio'] <- ratio
        
        K <- ceiling((ratio * max(rowSums(data$y)) + 10) / params$rho.gen[[1]])
      }
      
      result[i.method, 'K'] <- K
      
      # begin timing
      time.start <- proc.time()[3]
      
      fit <- pcountOpen(~1, ~1, ~1, ~1,
                        umf,
                        K = K,
                        immigration = params$immigration,
                        dynamics    = params$dynamics)
      
      time.elapsed <- proc.time()[3] - time.start
      
    # use the rgdual (new) method
    } else if (grepl('pco_rgd', method)) {
      # begin timing
      time.start <- proc.time()[3]
      
      fit <- pcountOpen_rgdual(~1, ~1, ~1, ~1,
                               umf,
                               immigration = params$immigration,
                               dynamics    = params$dynamics)
      
      time.elapsed <- proc.time()[3] - time.start
    }
    
    # record the results
    if (any(c('all', 'rt') %in% params$response.vars))
      result[i.method, 'rt'] <- time.elapsed
    if (any(c('all', 'nll') %in% params$response.vars))
      result[i.method, 'nll'] <- fit@negLogLike
    if (any(c('all', 'n.iters') %in% params$response.vars))
      result[i.method, 'n.iters'] <- fit@opt$counts['gradient']
    if (any(c('all', 'n.evals') %in% params$response.vars))
      result[i.method, 'n.evals'] <- fit@opt$counts['function']
    if (any(c('all', 'fit', 'fit.lambda') %in% params$response.vars) && !is.null(fit["lambda"]))
      result[i.method, 'fit.lambda'] <- coef(backTransform(fit, "lambda"))
    if (any(c('all', 'fit', 'fit.gamma') %in% params$response.vars) && !is.null(fit["gamma"]))
      result[i.method, 'fit.gamma'] <- coef(backTransform(fit, "gamma"))
    if (any(c('all', 'fit', 'fit.omega') %in% params$response.vars) && !is.null(fit["omega"]))
      result[i.method, 'fit.omega'] <- coef(backTransform(fit, "omega"))
    if (any(c('all', 'fit', 'fit.rho') %in% params$response.vars) && !is.null(fit["det"]))
      result[i.method, 'fit.rho'] <- coef(backTransform(fit, "det"))
    if (any(c('all', 'fit', 'fit.iota') %in% params$response.vars) && !is.null(fit["iota"]))
      result[i.method, 'fit.iota'] <- coef(backTransform(fit, "iota"))
  }
  
  if(data.sampled)
    return(list('result' = result, 'data' = data))
  else
    return(result)
}

generate_data <- function(params = params.default()) {
  # utility function for parameters which "repicates" the last object of a list if indexing past the end of the list
  # this means all parameters can be expressed as scalars/primitives or as lists
  # if the list length is less than T, this function essentially copies the last element when k > length(list)
  list.lookup <- function(list, k) { list[[min(length(list), k)]] }
  
  # lookup the sampling function (rather, create lookup functions)
  arrival.distns   <- list(poisson = rpois,
                           negbin  = rnbinom,
                           logis   = rlogis,
                           none    = function(n, p) {array(0, length(n))})
  
  offspring.distns <- list(bernoulli = function(n, p) {rbinom(length(n), n, p)},
                           poisson   = function(n, p) {rpois(length(n),  n * p)},
                           geometric = function(n, p) {rgeom(length(n), n, p)},
                           none      = function(n, p) {array(0, length(n))})
  
  # init data stucts
  N <- array(NA, c(params$n.samples, params$M, params$T))
  y <- array(NA, c(params$n.samples, params$M, params$T))
  S <- array(NA, c(params$n.samples, params$M, params$T - 1))
  G <- array(NA, c(params$n.samples, params$M, params$T - 1))
  A <- array(NA, c(params$n.samples, params$M, params$T))
  
  # sample data
  A[,,1] <- arrival.distns[[list.lookup(params$arrival.gen, 1)]](params$n.samples * params$M, list.lookup(params$iota.gen, 1))
  N[,,1] <- A[,,1]
  y[,,1] <- rbinom(N[,,1], N[,,1], list.lookup(params$rho.gen, 1))
  
  for(k in 2:params$T) {
    S[,,k-1] <- offspring.distns[[list.lookup(params$survival.gen,  k-1)]](N[,,k-1], list.lookup(params$omega.gen, k-1))
    G[,,k-1] <- offspring.distns[[list.lookup(params$offspring.gen, k-1)]](N[,,k-1], list.lookup(params$gamma.gen, k-1))
    A[,,k]   <- arrival.distns  [[list.lookup(params$arrival.gen,   k)]]  (N[,,k-1], list.lookup(params$iota.gen,  k))
    N[,,k]   <- S[,,k-1] + G[,,k-1] + A[,,k]
    
    y[,,k] <- rbinom(N[,,k], N[,,k], list.lookup(params$rho.gen, k))
  }
  
  # remove singleton dimensions
  N <- drop(N)
  y <- drop(y)
  S <- drop(S)
  G <- drop(G)
  A <- drop(A)
  
  data <- list(y = y,
               N = N,
               S = S,
               G = G,
               A = A,
               params = params)
  return(data)
}

# function for formating a number of seconds as HMS
s2hms <- function(s) {
  h <- floor(s / 3600)
  m <- floor((s - 3600*h) / 60)
  s <- floor(s - 3600*h - 60*m)
  
  if (h > 0)
    sprintf('%dh%02dm%02ds', h, m, s)
  else if (m > 0)
    sprintf('%dm%02ds', m, s)
  else
    sprintf('%ds', s)
}

write.params <- function(params, file.path) {
  file <- file(file.path)
  
  writeLines(mapply(function(name, val) {sprintf("%s: %s", name, toString(val))}, 
                    names(params), 
                    params),
             file)
  
  close(file)
}

result <- pco_rgd_experiment()
print(result$result)