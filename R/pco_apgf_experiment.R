# PCO vs PCO_APGFFWD comparison

# rm(list=ls())

library(unmarked)
source('pcountOpen_apgffwd.R')

library(abind)
library(beepr)
source('util.R')

nll_comparison_pco_vs_rgd <- function() {
  params <- params.default()
  
  params$methods <- c('pco_rgd', 'pco_trunc:default')
  
  params$M <- 1
  params$T <- 5
  
  params$n.iters <- 20
  params$shared.data <- 'none'
  
  params$experiment.variable <- 'rho.gen'
  # params$experiment.values   <- 0.95
  params$experiment.values   <- seq(0.05, 0.95, by=0.2)
  
  params$gamma.gen <- 0.2 # mean offspring
  params$omega.gen <- 0.5 # survival probability
  params$rho.gen   <- 0.8 # detection probability
  params$iota.gen  <- 20  # immigration rate
  
  result <- run_experiment(params = params)
  
  return(result)
}

params.default <- function() {
  params <- list()
  
  # params$methods <- c('pco_rgd')
  params$methods <- c('pco_rgd', 'pco_trunc:ratio')
  # params$methods <- c('pco_trunc:default', 'pco_trunc:ratio', 'pco_rgd') # which methods to test
  params$response.vars <- c('nll', 'rt', 'n.iters', 'fit', 'x')          # which values to measure/record
  
  params$M <- 3 # num of sites
  params$T <- 5 # num of timesteps
  
  params$n.samples <- 1  # num of samples to generate
  params$n.iters   <- 10  # num of times to replicate each trial
  params$n.attempts <- 3 # num of times to reattempt a failed trial
  
  params$shared.data <- 'trial' # {'trial', 'experiment', or 'none'}
                                # if 'experiment', then data is shared across all trials and all experiment values
                                # if 'trial', then data is shared between all trials
                                # if 'none', then new data is generated for each trial
  params$experiment.variable <- 'rho.gen' # the variable to be varied experimentally
  params$experiment.values   <- seq(0.05, 0.95, by=0.2)
  
  params$gamma.gen <- 0.2 # mean offspring
  params$omega.gen <- 0.5 # survival probability
  params$rho.gen   <- 0.8 # detection probability
  params$iota.gen  <- 10  # immigration rate
  
  # pco dynamics, one of {"constant", "autoreg", "notrend", "trend", "ricker", "gompertz"}
  # plus optionally immigration
  # these determine the fitted model structure
  params$dynamics    <- 'autoreg' 
  params$immigration <- TRUE
  
  # data generation distributions (note: these do not have to correspond to the dynamics/imm selected above)
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
  params.meta$experiment.name    <- "optim_eval"
  
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
  
  result <- experiment.fun(params = params, params.meta = params.meta)
  
  if (params.meta$write.results) {
    write.table(result, file = file.path(params.meta$results.path, 'results.df'))
    
    write.params(params,       file.path(params.meta$results.path, 'params.txt'))
    write.params(params.meta,  file.path(params.meta$results.path, 'params.meta.txt'))
  }
  
  return(list(result=result,
              params      = params,
              params.meta = params.meta))
}

plot_result <- function(result          = NULL,
                        result.dir      = NULL,
                        params.analysis = params.analysis.default()) {
  
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
        cat(sprintf("Trial #%d, %s = %s...\n", i.trial, params$experiment.variable, toString(experiment.value)))
      
      # generate trial-specific data (shared only across the different methods)
      if (params$shared.data == 'none')
        data <- generate_data(params.trial)
      
      i.attempt <- 1
      while(i.attempt < params$n.attempts) {
        if(i.attempt > 1 && !params.meta$silent)
          cat(sprintf("attempt %d...", i.attempt))
        
        withCallingHandlers({
          # run the trial
          result.trial <- pco_apgf_trial(params.trial, data, params.meta$silent)
          
          # add the experiment value and reorder the columns
          result.trial[,'n.attempts'] <- i.attempt
          result.trial[,params$experiment.variable] <- experiment.value
          result.trial <- result.trial[,c(ncol(result.trial), 1:(ncol(result.trial) - 1))]
          
          # append this trial to the record
          result <- rbind(result, result.trial)
          
          break
        }, error = function(e) {
          if (!params.meta$silent)
            cat(sprintf(' Error: %s.\n', e))
          
          next
        })
      } #/loop over attempts
      
      # print the runtime, approximate remaining time
      overall.elapsed.time <- proc.time()[3] - start.time
      percent.completed <- ((((i.experiment.value - 1) * params$n.iters) + i.trial) / 
                              (length(params$experiment.values) * params$n.iters))
      expected.remaining.time <- overall.elapsed.time * (1 - percent.completed) / percent.completed
      
      if (!params.meta$silent)
        cat(sprintf('remaining time ~= %s\n\n', s2hms(expected.remaining.time)))
    } #/loop over trials
  } #/loop over experiment values
  
  return(result)
}

pco_apgf_trial <- function(params = params.default(), 
                          data   = NULL,
                          silent = TRUE) {
  # generate data
  if(is.null(data)) {
    stopifnot(is.null(params$n.samples) || params$n.samples == 1)
    params$n.samples <- 1
    data <- generate_data(params)
    data.sampled = TRUE
  } else
    data.sampled = FALSE
  
  if(!silent)
    cat(sprintf('data = %s\n', mat2str(data$y)))
  
  # build an unmarkedframe
  umf <- unmarkedFramePCO(y = data$y, numPrimary = params$T)

  # prepare results
  result <- data.frame()
  for (i.method in 1:length(params$methods)) {
    method <- params$methods[[i.method]]
    
    result[i.method, 'method'] <- method
    
    if(!silent)
      cat(sprintf('  method = %s', method))
    
    # use the truncated (original) method
    if (grepl('pco_trunc', method)) {
      # set K
      if ('K' %in% names(params) && params$K > max(data$y)) {
        K <- params$K
      } else if(grepl('default', method)) {
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
      
      fit <- pcountOpen_apgffwd(~1, ~1, ~1, ~1,
                               umf,
                               immigration = params$immigration,
                               dynamics    = params$dynamics)
      
      time.elapsed <- proc.time()[3] - time.start
    }
    
    if(!silent)
      cat(sprintf(', rt = %s\n', s2hms(time.elapsed)))
    
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
    if (any(c('all', 'x', 'x.lambda') %in% params$response.vars) && !is.null(fit["lambda"]))
      result[i.method, 'x.lambda'] <- coef(fit, "lambda")
    if (any(c('all', 'x', 'x.gamma') %in% params$response.vars) && !is.null(fit["gamma"]))
      result[i.method, 'x.gamma'] <- coef(fit, "gamma")
    if (any(c('all', 'x', 'x.omega') %in% params$response.vars) && !is.null(fit["omega"]))
      result[i.method, 'x.omega'] <- coef(fit, "omega")
    if (any(c('all', 'x', 'x.rho') %in% params$response.vars) && !is.null(fit["det"]))
      result[i.method, 'x.rho'] <- coef(fit, "det")
    if (any(c('all', 'x', 'x.iota') %in% params$response.vars) && !is.null(fit["iota"]))
      result[i.method, 'x.iota'] <- coef(fit, "iota")
    
    # record the data
    result[i.method, 'y'] <- mat2str(data$y)
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
  arrival.distns   <- list(poisson = function(size, theta) {rpois  (n = size, lambda = theta)},
                           negbin  = rnbinom,
                           logis   = rlogis,
                           none    = function(size, theta=0) {array(0, size)})
  
  offspring.distns <- list(bernoulli = function(n, theta)   {rbinom(n = length(n), size   = n,        prob = theta)},
                           poisson   = function(n, theta)   {rpois (n = length(n), lambda = n * theta)},
                           geometric = function(n, theta)   {rgeom (n = length(n), prob   = theta)},
                           none      = function(n, theta=0) {array(0, length(n))})
  
  # init data stucts
  N <- array(NA, c(params$n.samples, params$M, params$T))
  y <- array(NA, c(params$n.samples, params$M, params$T))
  S <- array(NA, c(params$n.samples, params$M, params$T - 1))
  G <- array(NA, c(params$n.samples, params$M, params$T - 1))
  A <- array(NA, c(params$n.samples, params$M, params$T))
  
  # sample data
  A[,,1] <- arrival.distns[[list.lookup(params$arrival.gen, 1)]](params$n.samples * params$M, list.lookup(params$iota.gen, 1))
  N[,,1] <- A[,,1]
  y[,,1] <- rbinom(n    = length(N[,,1]), 
                   size = N[,,1], 
                   prob = list.lookup(params$rho.gen, 1))
  
  for(k in 2:params$T) {
    S[,,k-1] <- offspring.distns[[list.lookup(params$survival.gen,  k-1)]](N[,,k-1], list.lookup(params$omega.gen, k-1))
    G[,,k-1] <- offspring.distns[[list.lookup(params$offspring.gen, k-1)]](N[,,k-1], list.lookup(params$gamma.gen, k-1))
    A[,,k]   <- arrival.distns  [[list.lookup(params$arrival.gen,   k)]]  (params$M, list.lookup(params$iota.gen,  k))
    N[,,k]   <- S[,,k-1] + G[,,k-1] + A[,,k]
    
    y[,,k] <- rbinom(n    = length(N[,,k]), 
                     size = N[,,k], 
                     prob = list.lookup(params$rho.gen, k))
  }
  
  # remove the first (n.samples) dimension (if it has extent 1)
  N <- adrop(N, 1)
  y <- adrop(y, 1)
  S <- adrop(S, 1)
  G <- adrop(G, 1)
  A <- adrop(A, 1)
  
  data <- list(y = y,
               N = N,
               S = S,
               G = G,
               A = A,
               params = params)
  return(data)
}

# function to parse a simple numeric matrix to a simple string form
mat2str <- function(x) {
  if (nrow(x) == 1)
    return(paste('[', toString(x), ']', sep=''))
  else
    return(paste('[[', paste(apply(x, 1, toString), collapse = '];['), ']]', sep=''))
}

# function to parse a standard numeric vector in string form
str2vec <- function(str) {
  str <- trimws(str)
  
  # remove leading/trailing brackets
  str <- regmatches(str, regexpr('(?=\\[*)[^\\[].*[^\\]](?=\\]*)', str, perl=TRUE))
  
  # convert to numeric vector
  str <- as.numeric(strsplit(str, ',')[[1]])
  
  # convert to a 2D array
  dim(str) <- c(1, length(str))
  
  return(str)
}

write.params <- function(params, file.path) {
  file <- file(file.path)
  
  writeLines(mapply(function(name, val) {sprintf("%s: %s", name, toString(val))}, 
                    names(params), 
                    params),
             file)
  
  close(file)
}

# result <- run_experiment()
# print(result$result)

result <- nll_comparison_pco_vs_rgd()
print(result$result)
beep()

# result.eval <- result$result[5,]
# y.eval      <- str2vec(result.eval$y)
# theta.eval  <- as.numeric(result.eval[11:15])
# 
# umf.eval    <- unmarkedFramePCO(y = y.eval, numPrimary = ncol(y.eval))
# nll.rgd     <- pcountOpen_rgdual(~1, ~1, ~1, ~1, data=umf.eval, mixture=result$params$mixture, dynamics=result$params$dynamics, immigration=result$params$immigration, eval.at=theta.eval,
#                                  nll.fun = 'rgd')
# 
# nll.pco_def <- pcountOpen_rgdual(~1, ~1, ~1, ~1, data=umf.eval, mixture=result$params$mixture, dynamics=result$params$dynamics, immigration=result$params$immigration, eval.at=theta.eval,
#                                  nll.fun = 'trunc')
# 
# nll.pco_safe<- pcountOpen_rgdual(~1, ~1, ~1, ~1, data=umf.eval, mixture=result$params$mixture, dynamics=result$params$dynamics, immigration=result$params$immigration, eval.at=theta.eval,
#                                  nll.fun = 'trunc', K = 500)

# result.eval2 <- result$result[2,]
# y.eval2      <- str2vec(result.eval2$y)
# theta.eval2  <- as.numeric(result.eval2[11:15])
# 
# umf.eval2    <- unmarkedFramePCO(y = y.eval2, numPrimary = ncol(y.eval2))
# nll.rgd2     <- pcountOpen_rgdual(~1, ~1, ~1, ~1, data=umf.eval2, mixture=result$params$mixture, dynamics=result$params$dynamics, immigration=result$params$immigration, eval.at=theta.eval2,
#                                  nll.fun = 'rgd')
# 
# nll.pco_def2 <- pcountOpen_rgdual(~1, ~1, ~1, ~1, data=umf.eval2, mixture=result$params$mixture, dynamics=result$params$dynamics, immigration=result$params$immigration, eval.at=theta.eval2,
#                                  nll.fun = 'trunc')
# 
# nll.pco_safe2<- pcountOpen_rgdual(~1, ~1, ~1, ~1, data=umf.eval2, mixture=result$params$mixture, dynamics=result$params$dynamics, immigration=result$params$immigration, eval.at=theta.eval2,
#                                  nll.fun = 'trunc', K = 250)

# result.eval3 <- result$result[1,]
# y.eval3      <- str2vec(result.eval3$y)
# theta.eval3  <- as.numeric(result.eval3[11:15])
# 
# umf.eval3    <- unmarkedFramePCO(y = y.eval3, numPrimary = ncol(y.eval3))
# nll.rgd3     <- pcountOpen_rgdual(~1, ~1, ~1, ~1, data=umf.eval3, mixture=result$params$mixture, dynamics=result$params$dynamics, immigration=result$params$immigration, eval.at=theta.eval3,
#                                   nll.fun = 'rgd')
# 
# nll.pco_def3 <- pcountOpen_rgdual(~1, ~1, ~1, ~1, data=umf.eval3, mixture=result$params$mixture, dynamics=result$params$dynamics, immigration=result$params$immigration, eval.at=theta.eval3,
#                                   nll.fun = 'trunc')
# 
# nll.pco_safe3<- pcountOpen_rgdual(~1, ~1, ~1, ~1, data=umf.eval3, mixture=result$params$mixture, dynamics=result$params$dynamics, immigration=result$params$immigration, eval.at=theta.eval3,
#                                   nll.fun = 'trunc', K = 250)

# theta.eval4 <- theta.eval3
# theta.eval4[c(1,4,5)] <- theta.eval4[c(1,4,5)] / 3.
# nll.rgd4     <- pcountOpen_rgdual(~1, ~1, ~1, ~1, data=umf.eval3, mixture=result$params$mixture, dynamics=result$params$dynamics, immigration=result$params$immigration, eval.at=theta.eval4,
#                                   nll.fun = 'rgd')
# 
# nll.pco_def4 <- pcountOpen_rgdual(~1, ~1, ~1, ~1, data=umf.eval3, mixture=result$params$mixture, dynamics=result$params$dynamics, immigration=result$params$immigration, eval.at=theta.eval4,
#                                   nll.fun = 'trunc')
# 
# nll.pco_safe4<- pcountOpen_rgdual(~1, ~1, ~1, ~1, data=umf.eval3, mixture=result$params$mixture, dynamics=result$params$dynamics, immigration=result$params$immigration, eval.at=theta.eval4,
#                                   nll.fun = 'trunc', K = 250)

# est.rgd4<- pcountOpen_rgdual(~1, ~1, ~1, ~1, data=umf.eval3, mixture=result$params$mixture, dynamics=result$params$dynamics, immigration=result$params$immigration,
#                                   nll.fun = 'rgd')
# est.pco_def4<- pcountOpen_rgdual(~1, ~1, ~1, ~1, data=umf.eval3, mixture=result$params$mixture, dynamics=result$params$dynamics, immigration=result$params$immigration,
#                                   nll.fun = 'trunc')
# est.pco_safe4<- pcountOpen_rgdual(~1, ~1, ~1, ~1, data=umf.eval3, mixture=result$params$mixture, dynamics=result$params$dynamics, immigration=result$params$immigration,
#                                   nll.fun = 'trunc', K = 250)

beep()