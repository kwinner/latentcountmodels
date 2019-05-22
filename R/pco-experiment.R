if (!suppressWarnings(require('unmarked', quietly = TRUE))) install.packages('unmarked'); library('unmarked')
if (!suppressWarnings(require('abind',    quietly = TRUE))) install.packages('abind');    library('abind')

source('experiment.R')
source('pcountOpen_rgdual.R')

# arrival definitions
arrival.distns   <- list(poisson = function(theta) {rpois  (n = 1, lambda = theta)},
                         negbin  = rnbinom, # TODO: UPDATE
                         logis   = rlogis,  # TODO: UPDATE
                         none    = function(theta = 0) {0})

# offspring definitions
offspring.distns <- list(bernoulli = function(n, theta)   {rbinom(n = 1, size   = n,        prob = theta)},
                         poisson   = function(n, theta)   {rpois (n = 1, lambda = n * theta)},
                         geometric = function(n, theta)   {rgeom (n = 1, prob   = theta)},
                         none      = function(n=0, theta=0) {0})

params.pco.default <- function() {
	params <- params.default()

	# the control variables
	# if length(experiment.var) == 1, then experiment.values should be a list
	# if length(experiment.var) > 1,  then experiment.values should be a 2D matrix 
	#   with ncol(experiment.values) == length(experiment.var)
	# for a grid search, you can use `expand.grid(list(paramA.values, paramB.values, ...))`
	params$experiment.var    <- 'rho.gen' # the variable (or variables) to be varied experimentally
	params$experiment.values <- seq(0.05, 0.95, by=0.10)

	# what methods to evaluate in each trial
	params$methods <- c('pco_rgd', 'pco_trunc:default')
	# which values to measure/record
	#   the first in the list will be considered the "primary" response variable for status reports
	params$response.vars <- c('nll', 'rt', 'n.iters', 'fit', 'x')

	# data shape parameters
	params$M <- 2  # number of independent sites
	params$T <- 5  # number of observations
	
	# pco dynamics, one of {"constant", "autoreg", "notrend", "trend", "ricker", "gompertz"}
	# plus optionally immigration
	# these determine the fitted model structure
	params$dynamics    <- 'trend' 
	params$immigration <- TRUE
	
	params$arrival.gen   <- 'poisson'
	params$survival.gen  <- 'bernoulli'
	params$offspring.gen <- 'poisson'
	
	# model parameters
	params$rho   <- 0.8 # detection probability
	params$gamma <- 0.2 # offspring params
	params$omega <- 0.5 # survival params
	params$iota  <- 20  # immigration params
	
	return(params)
} # /params.pco.default

# generate data from an open metapopulation w/ arrival, offspring, and immigration
# required parameters: (may optionally be suffixed by '.gen')
#   M             : number of independent sites
#   T             : number of observations
# note: all of the following parameters may given as scalars or vectors
#  if scalar or if the vector length is less than T, then the last element will be replicated as needed
#   arrival       : name of arrival distribution   {'poisson', 'negbin', 'logis', 'none'}
#   survival      : name of survival distribution  {'bernoulli', 'poisson', 'geometric', 'none'}, generally 'bernoulli' or 'none'
#   offspring     : name of offspring distribution {'bernoulli', 'poisson', 'geometric', 'none'}
#   rho           : detection probability
#   gamma         : offspring distribution parameters
#   omega         : survival distribution parameters
#   iota          : imigration distribution parameters
# optional parameter return.val for debugging purposes, can be set to any subset of {'y', 'N', 'S', 'G', 'A'} or 'all'
#   if more than one selected, the data will be returned in a named list
generate_data.pco <- function(params = params.pco.default(), return.val = 'y') {
  # if parameters have a .gen suffix, use those
  params.gen <- regmatches(names(params), regexpr('.*\\.gen$', names(params)))
  params[sub('\\.gen', '', params.gen)] <- params[params.gen]
  
  # if any of the params (besides M, T) are scalar or are not full-length, repeat the last entry to fill the parameter vector
  params$arrival   <- extend.parameter(params$arrival,   c(params$M, params$T))
  params$survival  <- extend.parameter(params$survival,  c(params$M, params$T - 1))
  params$offspring <- extend.parameter(params$offspring, c(params$M, params$T - 1))
  params$rho       <- extend.parameter(params$rho,       c(params$M, params$T))
  params$gamma     <- extend.parameter(params$gamma,     c(params$M, params$T - 1))
  params$omega     <- extend.parameter(params$omega,     c(params$M, params$T - 1))
  params$iota      <- extend.parameter(params$iota,      c(params$M, params$T))
  
  # init data stucts
  data <- list()
  data$N <- array(NA, c(params$M, params$T))     # N = latent count at times t in T
  data$y <- array(NA, c(params$M, params$T))     # y = observed count at times t in T
  data$S <- array(NA, c(params$M, params$T - 1)) # S = latent survivors from t to t+1
  data$G <- array(NA, c(params$M, params$T - 1)) # G = latent offspring from t to t+1
  data$A <- array(NA, c(params$M, params$T))     # A = latent arrivals at times t in T
  
  # sample data
  # t_0 is arrivals only
  data$A[,1] <- mapply(function(distn, p) {arrival.distns[[distn]](p)}, params$arrival[,1], params$iota[,1])
  data$N[,1] <- data$A[,1]
  data$y[,1] <- rbinom(n    = length(data$N[,1]),
                       size = data$N[,1],
                       prob = params$rho[,1])
  
  # begin the main loop
  for(k in 2:params$T) {
    data$S[,k-1] <- mapply(function(distn, n, p) {offspring.distns[[distn]](n, p)}, params$survival [,1], data$N[,k-1], params$omega[,1])
    data$G[,k-1] <- mapply(function(distn, n, p) {offspring.distns[[distn]](n, p)}, params$offspring[,1], data$N[,k-1], params$gamma[,1])
    data$A[,k]   <- mapply(function(distn,    p) {  arrival.distns[[distn]](p)},    params$arrival  [,1],               params$iota [,1])
    data$N[,k]   <- data$S[,k-1] + data$G[,k-1] + data$A[,k]
    
    data$y[,k] <- rbinom(n    = length(data$y[,k]),
                         size = data$N[,k],
                         prob = params$rho[,k])
  }
  
  if ('all' %in% return.val)
    return(data)
  else
    return(data[return.val])
} # /generate_data.pco

# extend a scalar, list, or matrix by copying the last element in each dimension out
extend.parameter <- function(param, shape) {
  # extending arrays of arbitrary dimension was messy and I don't intend to use it, so stick to 1 or 2 dimensions
  stopifnot(length(shape) <= 2)
  if (is.list(param) || is.vector(param)) {
    # param is a scalar or unidimensional
    if (length(shape) == 1) {
      # result is unidimensional as well
      if (shape <= length(param))
        # truncate (not really an intended use...)
        return(param[1:shape])
      else
        # clone param[-1] to fill out the vector
        return(c(param, rep(param[length(param)], shape - length(param))))
    } else { 
      # extend a unidimensional param to a matrix
      #  note: there is some ambiguity in whether param is a row vector or a column vector
      #  in my use cases, param is always a row vector, but to extend a column vector, 
      #  just convert it to an actual matrix first and it will fall into the case below
      
      # from the unidimensional case above
      if (shape[2] <= length(param))
        param <- param[1:shape[2]]
      else
        param <- c(param, rep(param[length(param)], shape[2] - length(param)))
      
      # now extend to a matrix
      return(matrix(rep(param, shape[1]), shape[1], shape[2], byrow=TRUE))
    }
  } else if (is.matrix(param)) { # there's a way to do arrays, but it's too much hassle
    stopifnot(length(shape) == 2) # this method is not meant to dimension squeeze
    
    result <- array(NA, shape)
    
    # copy any aligned parts of param to result
    overlap <- mapply(min, dim(param), dim(result))
    result[1:overlap[1], 1:overlap[2]] <- param[1:overlap[1], 1:overlap[2]]
    
    # extend the rows
    if (shape[1] > overlap[1])
      result[(overlap[1]+1):shape[1],] <- matrix(rep(result[overlap[1],],shape[1]-overlap[1]),shape[1]-overlap[1], shape[2], byrow=TRUE)
    
    # extend the cols
    if (shape[2] > overlap[2])
      result[,(overlap[2]+1):shape[2]] <- matrix(rep(result[,overlap[2]],shape[2]-overlap[2]),shape[1], shape[2] - overlap[2])
    
    return(result)
  }
} # /extend.paramter

trial.pco.fit <- function(data, params = params.pco.default()) {
  # build an unmarkedframe
  # siteCovs <- data.frame(rand = runif(params$M, min=0, max=1))
  # umf <- unmarkedFramePCO(y = data$y, numPrimary = params$T, siteCovs = siteCovs)
  umf <- unmarkedFramePCO(y = data$y, numPrimary = params$T)
  
  # prepare results
  result <- data.frame()
  for (i.method in 1:length(params$methods)) {
    # select and record the method (the algorithm/process for performing inference)
    method <- params$methods[[i.method]]
    result[i.method, 'method'] <- method
    if (params$verbose >= 3)
      cat(sprintf('    method: %s', method))
    
    # begin timing, then select one of the methods
    time.start <- proc.time()[3]
    if (grepl('pco_trunc', method)) {
      # set K
      if ('K' %in% names(params) && params$K > max(data$y)) {
        K <- params$K
      } else if(grepl('default', method)) {
        K <- max(data$y) + 20
      } else if (grepl('ratio', method)) {
        ratio <- 0.38563049853372433
        result[i.method, 'ratio'] <- ratio
        
        K <- ceiling((ratio * max(rowSums(data$y)) + 10) / params$rho[[1]])
      }
      result[i.method, 'K'] <- K
      
      fit <- pcountOpen(~1, ~1, ~1, ~1,
                        umf,
                        K = K,
                        immigration = params$immigration,
                        dynamics    = params$dynamics)
    } else if (grepl('pco_rgd', method)) {
      fit <- pcountOpen_rgdual(~1, ~1, ~1, ~1,
                               umf,
                               immigration = params$immigration,
                               dynamics    = params$dynamics)
    } # /method selection
    time.elapsed <- proc.time()[3] - time.start
    
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
    result[i.method, 'y'] <- mat2str(data$y) # this is a matrix/vector serialization script I wrote to do human readable matlab-format matrix print
    
    # print status
    if (params$verbose >= 3) {
      primary.response.var <- params$response.vars[1]
      
      # if the primary response var is all, default to rt
      # for rt, convert to human readable format
      if (any(c('all', 'rt') %in% primary.response.var))
        cat(sprintf(', rt = %s\n', s2hms(result[i.method, 'rt'])))
      else
        cat(sprintf(', %s = %s\n', primary.response.var, result[i.method, primary.response.var]))
    }
  } # /loop over methods
  
  return(result)
} # /trial.pco.fit

evaluate.result.nll <- function(result, params, method=NULL) {
  method = if (grepl('rgd', method)) 'rgd' else 'trunc'
  
  # select the cols containing the final parameter values from fitting
  eval.at <- as.numeric(result[grep('x', names(result))])
  
  umf <- unmarkedFramePCO(y = str2mat(result[['y']]), numPrimary = params$T)
  
  nll <- pcountOpen_rgdual(~1, ~1, ~1, ~1,
                           umf,
                           immigration = params$immigration,
                           dynamics    = params$dynamics,
                           nll.fun     = method,
                           eval.at     = eval.at)
  
  return(nll)
}

params <- params.pco.default()
params$dynamics <- "autoreg"
params$experiment.var <- 'iota'
params$experiment.values <- c(10, 50, 100, 200)
# params$experiment.var <- 'M'
# params$experiment.values <- c(2,4,6,8,10)
params$M <- 2
params$T <- 2
params$n.replications <- 10
result.dir <- experiment(trial.pco.fit, generate_data.pco, params)