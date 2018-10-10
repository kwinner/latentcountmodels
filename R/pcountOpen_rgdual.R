source("rgdual.R")

library(unmarked)

pcountOpen_rgdual <- function(lambdaformula, gammaformula, omegaformula, pformula,
                              data, 
                              mixture     = 'P', 
                              K, 
                              dynamics    = 'constant',
                              fix         = 'none',
                              starts, 
                              method      = 'BFGS', 
                              se          = TRUE, 
                              immigration = FALSE, 
                              iotaformula = ~1,
                              eval.at     = NULL,
                              nll.fun     = 'rgd',
                              ...) {
  mixture  <- match.arg(mixture,  c('P', 'NB', 'ZIP'))
  dynamics <- match.arg(dynamics, c('constant', 'autoreg', 'notrend', 'trend', 'ricker', 'gompertz'))
  fix      <- match.arg(fix,      c('none', 'gamma', 'omega'))
  nll.fun  <- match.arg(nll.fun,  c('auto', 'rgd', 'trunc'))
  
  #TODO: unused error handling from unmarked, safe to remove?
  ## if(identical(dynamics, "notrend") &
  ##    !identical(lambdaformula, omegaformula))
  ##     stop("lambdaformula and omegaformula must be identical for notrend model")
  
  if((identical(dynamics, "constant") || identical(dynamics, "notrend")) & immigration)
    stop("You can not include immigration in the constant or notrend models")
  
  # compile formula inputs, assemble into a design object (an unmarked concept)
  formlist <- list(lambdaformula=lambdaformula, gammaformula=gammaformula,
                   omegaformula=omegaformula, pformula=pformula, iotaformula=iotaformula)
  formula  <- as.formula(paste(unlist(formlist), collapse=" "))
  D <- getDesign(data, formula)
  
  #compute dimensionality
  M <- nrow(D$y)       #M = number of sites
  T <- data@numPrimary #T = number of observations
  J <- ncol(D$y) / T   #J = number of repeated counts
  
  #begin tracking whether we can use nll_rgd (use_rgdual = T) or if we need to fall back on nll_um
  use_rgdual = TRUE
  # if(J > 1)
  #   use_rgdual = FALSE
  
  #TODO: we can probably support these, but don't at the moment
  if(mixture == "NB" || mixture == "ZIP")
    use_rgdual = FALSE
  # these we likely cannot support
  if(dynamics == "ricker" || dynamics == "gompertz")
    use_rgdual = FALSE
  
  if(identical(nll.fun, 'rgd') && !use_rgdual)
    stop("Unable to evaluate model using rgdual.")
  
  # if possible, use rgdual unless trunc specified
  use_rgdual <- use_rgdual && !identical(nll.fun, 'trunc')
  
  # update the design matrix to handle missing data/missing parmeters
  D <- pcountOpen_handle_missing(D, M, T, J)
  
  # process the parameters specified in the design matrix
  D <- pcountOpen_parameter_structure(D, dynamics, immigration, mixture, fix)
  
  if(!missing(starts) && length(starts) != D$nP)
    stop(paste("The number of starting values should be", D$nP))
  
  D$ym <- matrix(D$y, nrow=M)
  
  # set some parameters related to the truncation
  if(use_rgdual == FALSE) {
    # K is the support truncation parameter for abundance
    if(missing(K))
      K <- K.default(D$y)
    
    if(K <= max(D$y, na.rm = TRUE))
      stop("specified K is too small. Try a value larger than any observation")
    
    k  <- 0:K
    lk <- length(k)
    
    #TODO: describe these indices...
    # Create indices (should be written in C++)
    I <- cbind(rep(0:K, times=K+1),
               rep(0:K, each=K+1))
    I1 <- I[I[,1] <= I[,2],]
    Ib <- Ip <- list()
    for(i in 1:nrow(I)) {
      Z <- 0:min(I[i,])
      Ib[[i]] <- which((I1[,1] %in% Z) & (I1[,2]==I[i,1])) - 1
      Ip[[i]] <- as.integer(I[i,2]-Z)
    }
  } else { # use_rgdual == TRUE
    K <- NaN
  }
  
  if(use_rgdual == FALSE) {
    # the previous pcount_open objective function, for when rgdual is unsupported
    objective.fun <- function(parms, debug=FALSE) {
      beta.lam  <- parms[1                          : D$nAP]
      beta.gam  <- parms[(D$nAP+1)                  :(D$nAP+D$nGP)]
      beta.om   <- parms[(D$nAP+D$nGP+1)            :(D$nAP+D$nGP+D$nOP)]
      beta.p    <- parms[(D$nAP+D$nGP+D$nOP+1)      :(D$nAP+D$nGP+D$nOP+D$nDP)]
      beta.iota <- parms[(D$nAP+D$nGP+D$nOP+D$nDP+1):(D$nAP+D$nGP+D$nOP+D$nDP+D$nIP)]
      log.alpha <- 1
      if(mixture %in% c("NB", "ZIP"))
        log.alpha <- parms[D$nP]
      .Call("nll_pcountOpen",
            D$ym,
            D$Xlam, D$Xgam, D$Xom, D$Xp, D$Xiota,
            beta.lam, beta.gam, beta.om, beta.p, beta.iota, log.alpha,
            D$Xlam.offset, D$Xgam.offset, D$Xom.offset, D$Xp.offset, D$Xiota.offset,
            D$ytna, D$yna,
            K+1, 
            mixture, 
            D$first, D$last, 
            M, J, T,
            D$delta, 
            dynamics, 
            fix, 
            D$go.dims, 
            immigration,
            I, I1, Ib, Ip,
            PACKAGE = "unmarked")
    }
  } else { # use_rgdual == TRUE
    # the objective function based on rgdual
    objective.fun <- function(parms, debug=FALSE) {
      # parameter extraction (following model from nll_pcountOpen.cpp, which does this in C)
      
      # lambda = initial arrival
      if(D$nAP > 0) {
        beta.lam <- parms[1:D$nAP]
        
        # backtransform lambda
        lambda <- as.vector(exp(D$Xlam * beta.lam + D$Xlam.offset))
      }
      # omega = survivorship rate
      if(D$nOP > 0) {
        beta.om <- parms[(D$nAP+D$nGP+1):(D$nAP+D$nGP+D$nOP)]
        
        # backtransform omega
        if(fix != "omega" && dynamics != "trend") {
          if(dynamics == "ricker" || dynamics == "gompertz") {
            omega <- as.vector(exp(D$Xom * beta.om + D$Xom.offset))
          } else {
            omega <- as.vector(logistic(D$Xom * beta.om + D$Xom.offset))
          }
        }
      }
      # gamma = recruitment rate
      if(D$nGP > 0) {
        beta.gam <- parms[(D$nAP+1):(D$nAP+D$nGP)]
        
        # backtransform gamma
        if(dynamics == "notrend") {
          #TODO: notrend support
          # gamma <- (1-om) % lamMat
        } else {
          if(fix != "gamma") {
            gamma <- as.vector(exp(D$Xgam * beta.gam + D$Xgam.offset))
          }
        }
      }
      # p = detection rate (also called det)
      if(D$nDP > 0) {
        beta.p <- parms[(D$nAP+D$nGP+D$nOP+1):(D$nAP+D$nGP+D$nOP+D$nDP)]
        
        # backtransform p
        det <- as.vector(logistic(D$Xp * beta.p + D$Xp.offset))
      }
      # iota = immigration rate
      if(D$nIP > 0) {
        beta.iota <- parms[(D$nAP+D$nGP+D$nOP+D$nDP+1):(D$nAP+D$nGP+D$nOP+D$nDP+D$nIP)]
        
        # transform iota
        iota <- as.vector(exp(D$Xiota * beta.iota + D$Xiota.offset))
      }
      # log.alpha = dispersion
      log.alpha <- 1
      if(mixture %in% c("NB", "ZIP")) {
        log.alpha <- parms[D$nP]
        
        # transform log alpha
        if(mixture == "NB")
          alpha <- exp(log.alpha)
        else #if(mixture == "ZIP")
          alpha <- logistic(log.alpha)
      }
      
      #### A note on rgdual-style dynamics: ################################################################################
      ## our model has two types of dynamics: arrivals and offspring
      ## there are no explicit convolutions needed to combine the two types like in the `tpX` methods of nll_pcountOpen.cpp
      ##   all transition dynamics are assumed to occur simultaneously and independently
      ## the arrival dist'n describes a number of new individuals introduced to the population between observation events,
      ##   and is independent of the current size of the population
      ## the offspring dist'n describes, for each individual i in the population at time t_k, how many individuals i will
      ##   "produce" at time t_{k+1}. This count includes i if they survive to the next observation event
      ##   it is assumed implicitly that the offspring of each individual in the population at t_k are iid
      # select the arrival distribution (and params)
      if(dynamics %in% c("constant", "notrend")) {
        arrival.pgf <- pgf.poisson
        # initial arrivals are parameterized by lambda, then immigration is parameterized by gamma
        theta.arrival <- data.frame(lambda = c(lambda, array(gamma, T)))
      } else if(immigration) {
        arrival.pgf   <- pgf.poisson
        # initial arrivals are parameterized by lambda, then immigration is parameterized by iota
        theta.arrival <- data.frame(lambda = c(lambda, array(iota, T)))
      } else {
        # if no immigration, use the PGF for a uniform dist'n w/ all mass on 0
        arrival.pgf   <- pgf.zero
        theta.arrival <- data.frame(theta = array(0, T))
      }
      
      # select the offspring distribution
      if(dynamics %in% c("constant")) {
        offspring.pgf   <- pgf.bernoulli
        theta.offspring <- data.frame(p = array(omega, T - 1))
      } else if(dynamics %in% c("autoreg")) {
        offspring.pgf   <- function(s, theta) {
          return(pgf.bernoulli(s, theta['p']) * pgf.poisson(s, theta['lambda']))
        }
        theta.offspring <- data.frame(p      = array(omega, T - 1),
                                      lambda = array(gamma, T - 1))
      } else if(dynamics %in% c("trend")) {
        offspring.pgf   <- pgf.poisson
        theta.offspring <- data.frame(lambda = array(gamma, T - 1))
      }
      #TODO: notrend model
      
      # detection dist'n is always binomial
      theta.observ    <- data.frame(p=array(det, T))
      
      nll <- 0
      for (i.site in 1:M) {
        # at the moment our C wrappers are not overloaded for integers, so everything needs to be doubles
        # TODO: update C wrappers to take integers (as is, this cannot be done w/ automatic dispatch)
        ym_double <- as.double(D$ym[i.site,])
        Alpha <- forward(ym_double, arrival.pgf, theta.arrival, offspring.pgf, theta.offspring, theta.observ, d=0)
        if(debug) browser()
        
        nll <- nll - forward.ll(Alpha)
      }
      
      return(nll)
    }
  } # /if use_rgdual == FALSE
  
  # if eval.at provided and consistent, just call the objective function once and return
  if(!missing(eval.at) && !is.null(eval.at) && length(eval.at) == D$nP)
    return(objective.fun(eval.at))
  
  if(missing(starts))
    starts <- rep(0, D$nP)
  
  fm <- optim(starts, objective.fun, method=method, hessian=se, ...)
  # fm2 <- optim(starts, nll_um, method=method, hessian=se, ...)
  
  opt <- fm
  ests <- fm$par
  if(identical(mixture,"NB"))
    nbParm <- "alpha"
  else if(identical(mixture, "ZIP"))
    nbParm <- "psi"
  else
    nbParm <- character(0)
  names(ests) <- c(D$lamParms, D$gamParms, D$omParms, D$detParms, D$iotaParms, D$nbParm)
  if(se) {
    covMat <- tryCatch(solve(fm$hessian), error=function(x)
      simpleError("Hessian is not invertible. Try using fewer covariates or providing starting values."))
    if(class(covMat)[1] == "simpleError") {
      print(covMat$message)
      covMat <- matrix(NA, D$nP, D$nP)
    }
  } else covMat <- matrix(NA, D$nP, D$nP)
  
  fmAIC <- 2*fm$value + 2*D$nP
  
  lamEstimates <- unmarkedEstimate(name = "Abundance", short.name = "lam",
                                   estimates = ests[1:D$nAP], covMat = as.matrix(covMat[1:D$nAP,1:D$nAP]),
                                   invlink = "exp", invlinkGrad = "exp")
  detEstimates <- unmarkedEstimate(name = "Detection", short.name = "p",
                                   estimates = ests[(D$nAP+D$nGP+D$nOP+1) : (D$nAP+D$nGP+D$nOP+D$nDP)],
                                   covMat = as.matrix(covMat[(D$nAP+D$nGP+D$nOP+1) : (D$nAP+D$nGP+D$nOP+D$nDP),
                                                             (D$nAP+D$nGP+D$nOP+1) : (D$nAP+D$nGP+D$nOP+D$nDP)]),
                                   invlink = "logistic", invlinkGrad = "logistic.grad")
  estimateList <- unmarkedEstimateList(list(lambda=lamEstimates))
  gamName <- switch(dynamics,
                    constant = "gamConst",
                    autoreg = "gamAR",
                    notrend = "",
                    trend = "gamTrend",
                    ricker = "gamRicker",
                    gompertz = "gamGomp")
  if(!(identical(fix, "gamma") | identical(dynamics, "notrend")))
    estimateList@estimates$gamma <- unmarkedEstimate(name =
                                                       ifelse(identical(dynamics, "constant") | identical(dynamics, "autoreg"),
                                                              "Recruitment", "Growth Rate"), short.name = gamName,
                                                     estimates = ests[(D$nAP+1) : (D$nAP+D$nGP)], covMat = as.matrix(covMat[(D$nAP+1) :
                                                                                                                        (D$nAP+D$nGP), (D$nAP+1) : (D$nAP+D$nGP)]),
                                                     invlink = "exp", invlinkGrad = "exp")
  if(!(identical(fix, "omega") | identical(dynamics, "trend"))) {
    if(identical(dynamics, "constant") | identical(dynamics, "autoreg") | identical(dynamics, "notrend"))
      estimateList@estimates$omega <- unmarkedEstimate(
        name="Apparent Survival",
        short.name = "omega", estimates = ests[(D$nAP+D$nGP+1) :(D$nAP+D$nGP+D$nOP)],
        covMat = as.matrix(covMat[(D$nAP+D$nGP+1) : (D$nAP+D$nGP+D$nOP),
                                  (D$nAP+D$nGP+1) : (D$nAP+D$nGP+D$nOP)]),
        invlink = "logistic", invlinkGrad = "logistic.grad")
    else if(identical(dynamics, "ricker"))
      estimateList@estimates$omega <- unmarkedEstimate(
        name="Carrying Capacity",
        short.name = "omCarCap", estimates = ests[(D$nAP+D$nGP+1) :(D$nAP+D$nGP+D$nOP)],
        covMat = as.matrix(covMat[(D$nAP+D$nGP+1) : (D$nAP+D$nGP+D$nOP),
                                  (D$nAP+D$nGP+1) : (D$nAP+D$nGP+D$nOP)]),
        invlink = "exp", invlinkGrad = "exp")
    else
      estimateList@estimates$omega <- unmarkedEstimate(
        name="Carrying Capacity",
        short.name = "omCarCap", estimates = ests[(D$nAP+D$nGP+1) :(D$nAP+D$nGP+D$nOP)],
        covMat = as.matrix(covMat[(D$nAP+D$nGP+1) : (D$nAP+D$nGP+D$nOP),
                                  (D$nAP+D$nGP+1) : (D$nAP+D$nGP+D$nOP)]),
        invlink = "exp", invlinkGrad = "exp")
  }
  estimateList@estimates$det <- detEstimates
  if(immigration) {
    estimateList@estimates$iota <- unmarkedEstimate(
      name="Immigration",
      short.name = "iota", estimates = ests[(D$nAP+D$nGP+D$nOP+D$nDP+1) :(D$nAP+D$nGP+D$nOP+D$nDP+D$nIP)],
      covMat = as.matrix(covMat[(D$nAP+D$nGP+D$nOP+D$nDP+1) : (D$nAP+D$nGP+D$nOP+D$nDP+D$nIP),
                                (D$nAP+D$nGP+D$nOP+D$nDP+1) : (D$nAP+D$nGP+D$nOP+D$nDP+D$nIP)]),
      invlink = "exp", invlinkGrad = "exp")
  }
  if(identical(mixture, "NB")) {
    estimateList@estimates$alpha <- unmarkedEstimate(name = "Dispersion",
                                                     short.name = "alpha", estimates = ests[D$nP],
                                                     covMat = as.matrix(covMat[D$nP, D$nP]), invlink = "exp",
                                                     invlinkGrad = "exp")
  }
  if(identical(mixture, "ZIP")) {
    estimateList@estimates$psi <- unmarkedEstimate(name = "Zero-inflation",
                                                   short.name = "psi", estimates = ests[D$nP],
                                                   covMat = as.matrix(covMat[D$nP, D$nP]), invlink = "logistic",
                                                   invlinkGrad = "logistic.grad")
  }
  umfit <- new("unmarkedFitPCO", fitType = "pcountOpen",
               call = match.call(), formula = formula, formlist = formlist, data = data,
               sitesRemoved=D$removed.sites, estimates = estimateList, AIC = fmAIC,
               opt = opt, negLogLike = fm$value, nllFun = objective.fun, K = K, mixture = mixture,
               dynamics = dynamics, immigration = immigration)
  return(umfit)
}

# update a design matrix to handle missing params/data
pcountOpen_handle_missing <- function(D, M, T, J) {
  # handle missing parameters
  if(is.null(D$Xlam.offset))  D$Xlam.offset  <- rep(0, M)
  if(is.null(D$Xgam.offset))  D$Xgam.offset  <- rep(0, M*(T-1))
  if(is.null(D$Xom.offset))   D$Xom.offset   <- rep(0, M*(T-1))
  if(is.null(D$Xp.offset))    D$Xp.offset    <- rep(0, M*T*J)
  if(is.null(D$Xiota.offset)) D$Xiota.offset <- rep(0, M*(T-1))
  
  # handle missing data
  D$yna    <- is.na(D$y)
  D$yna[]  <- as.integer(D$yna)
  D$y      <- array(D$y, c(M, J, T))
  D$ytna   <- apply(is.na(D$y), c(1,3), all)
  D$ytna   <- matrix(D$ytna, nrow=M)
  D$ytna[] <- as.integer(D$ytna)
  
  # find first and last non-missing data entries
  D$first <- apply(!D$ytna, 1, function(x) min(which(x)))
  D$last  <- apply(!D$ytna, 1, function(x) max(which(x)))
  
  return(D)
}

# add some parameter vector structure information to the design matrix
pcountOpen_parameter_structure <- function(D, dynamics, immigration, mixture, fix = 'None') {
  # extract the names of the parameters
  D$lamParms  <- colnames(D$Xlam)
  D$gamParms  <- colnames(D$Xgam)
  D$omParms   <- colnames(D$Xom)
  D$detParms  <- colnames(D$Xp)
  D$iotaParms <- colnames(D$Xiota)
  
  # count the number of parameters
  D$nAP <- ncol(D$Xlam)
  D$nGP <- ncol(D$Xgam)
  D$nOP <- ncol(D$Xom)
  D$nDP <- ncol(D$Xp)
  D$nIP <- ncol(D$Xiota)
  
  # turn off immigration params if immigration == FALSE
  if(!immigration) {
    D$iotaParms <- character(0)
    D$nIP       <- 0
  }
  
  # if a parameter is being fixed, check that the right params are set
  if(identical(fix, 'gamma') || identical(fix, 'omega')) {
    if(!identical(dynamics, "constant")) stop("dynamics must be constant when fixing gamma or omega")
    if(identical(fix, 'gamma')) {
      if(D$nGP > 1) stop("gamma covariates not allowed when fix==gamma")
      
      D$nGP      <- 0
      D$gamParms <- character(0)
    } else if(identical(fix, 'omega')) {
      if(D$nOP > 1) stop("omega covariates not allowed when fix==omega")
      
      D$nOP     <- 0
      D$omParms <- character(0)
    }
  }
  
  # check that simpler models have the right params set
  if(identical(dynamics, "trend") && !identical(fix, 'omega')) {
    if(D$nOP > 1) stop("omega covariates not allowed when dynamics==trend")
    
    D$nOP     <- 0
    D$omParms <- character(0)
  }
  if(identical(dynamics, "notrend") && !identical(fix, 'gamma')) {
    if(D$nGP > 1) stop("gamma covariates not allowed when dynamics==notrend")
    
    D$nGP      <- 0
    D$gamParms <- character(0)
  }
  
  # total number of parameters (+1 if mixture includes log.alpha)
  D$nP <- D$nAP + D$nGP + D$nOP + D$nDP + D$nIP + (mixture == 'NB' || mixture == 'ZIP')
  
  return(D)
}

# set a K which is consistent with the data
K.default <- function(y) {
  #TODO: we've found that using the following scheme for setting K is significantly more robust:
  # Y <- max(y, na.rm=T)
  # scale <- 0.3856 # (determined empirically)
  # K <- ceiling((scale * Y + 10) / p)
  K <- max(y, na.rm=T) + 20
  #warning("K was not specified and was set to ", K, ".")
  return(K)
}

environment(pcountOpen_rgdual) <- environment(pcountOpen)
environment(pcountOpen_handle_missing) <- environment(pcountOpen)
environment(pcountOpen_parameter_structure) <- environment(pcountOpen)
environment(K.default) <- environment(pcountOpen)

# environment(pcountOpen_rgdual) <- as.environment('namespace:unmarked')
# environment(pcountOpen_handle_missing) <- as.environment('package:unmarked')
# environment(pcountOpen_parameter_structure) <- as.environment('package:unmarked')
# environment(K.default) <- as.environment('package:unmarked')