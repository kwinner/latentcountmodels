source("rgdual.R")

pcountOpen_rgdual <- function(lambdaformula, gammaformula, omegaformula, pformula,
                               data, mixture=c("P", "NB", "ZIP"), K, dynamics=c("constant", "autoreg",
                                                                                "notrend", "trend", "ricker", "gompertz"),
                               fix=c("none", "gamma", "omega"),
                               starts, method="BFGS", se=TRUE, immigration=FALSE, iotaformula=~1, ...)
{
  mixture  <- match.arg(mixture)
  dynamics <- match.arg(dynamics)
  fix      <- match.arg(fix)
  
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
  
  #TODO: what is go.dims?
  go.dims <- D$go.dims
  deltamax <- max(D$delta, na.rm=TRUE)
  
  # rename design elements for readability
  y     <- D$y
  Xlam  <- D$Xlam
  Xgam  <- D$Xgam
  Xom   <- D$Xom
  Xp    <- D$Xp
  Xiota <- D$Xiota
  delta <- D$delta
  
  #compute dimensionality
  M <- nrow(y)               #M = number of sites
  T <- data@numPrimary       #T = number of observations
  J <- ncol(getY(data)) / T  #J = number of repeated counts
  
  #begin tracking whether we can use nll_rgd (use_rgdual = T) or if we need to fall back on nll_um
  use_rgdual = TRUE
  if(J > 1)
    use_rgdual = FALSE
  
  Xlam.offset  <- D$Xlam.offset
  Xgam.offset  <- D$Xgam.offset
  Xom.offset   <- D$Xom.offset
  Xp.offset    <- D$Xp.offset
  Xiota.offset <- D$Xiota.offset
  if(is.null(Xlam.offset))  Xlam.offset  <- rep(0, M)
  if(is.null(Xgam.offset))  Xgam.offset  <- rep(0, M*(T-1))
  if(is.null(Xom.offset))   Xom.offset   <- rep(0, M*(T-1))
  if(is.null(Xp.offset))    Xp.offset    <- rep(0, M*T*J)
  if(is.null(Xiota.offset)) Xiota.offset <- rep(0, M*(T-1))
  
  # handle missing data
  yna    <- is.na(y)
  yna[]  <- as.integer(yna)
  y      <- array(y, c(M, J, T))
  ytna   <- apply(is.na(y), c(1,3), all)
  ytna   <- matrix(ytna, nrow=M)
  ytna[] <- as.integer(ytna)
  
  # find first and last non-missing data entries
  first <- apply(!ytna, 1, function(x) min(which(x)))
  last  <- apply(!ytna, 1, function(x) max(which(x)))
  
  #TODO: describe
  lamParms <- colnames(Xlam)
  gamParms <- colnames(Xgam)
  omParms  <- colnames(Xom)
  detParms <- colnames(Xp)
  
  # number of free parameters of each type
  nAP <- ncol(Xlam) # nAP := initial abundance params
  nGP <- ncol(Xgam) # nGP := growth params
  nOP <- ncol(Xom)  # nOP := decay params
  nDP <- ncol(Xp)   # nDP := detection params
                    # nIP := immig. params (set below)
  if(immigration) {
    iotaParms <- colnames(Xiota)
    nIP       <- ncol(Xiota)
  } else {
    iotaParms <- character(0)
    nIP       <- 0
  }
  
  # enforce parameter selection for specific models/fixed parameters (gamma or omega)
  if(identical(fix, "gamma")) {
    if(!identical(dynamics, "constant"))
      stop("dynamics must be constant when fixing gamma or omega")
    if(nGP > 1)
      stop("gamma covariates not allowed when fix==gamma")
    else {
      nGP <- 0
      gamParms <- character(0)
    }
  }
  else if(identical(dynamics, "notrend")) {
    if(nGP > 1)
      stop("gamma covariates not allowed when dyamics==notrend")
    else {
      nGP <- 0
      gamParms <- character(0)
    }
  }
  
  if(identical(fix, "omega")) {
    if(!identical(dynamics, "constant"))
      stop("dynamics must be constant when fixing gamma or omega")
    if(nOP > 1)
      stop("omega covariates not allowed when fix==omega")
    else {
      nOP <- 0
      omParms <- character(0)
    }
  } else if(identical(dynamics, "trend")) {
    if(nOP > 1)
      stop("omega covariates not allowed when dynamics='trend'")
    else {
      nOP <- 0
      omParms <- character(0)
    }
  }
  
  # total number of parameters (if mixture == "NB" or "ZIP", then one extra parameter)
  #TODO: why is this not included in nGP/nOP?
  nP <- nAP + nGP + nOP + nDP + nIP + (mixture!="P")
  if(!missing(starts) && length(starts) != nP)
    stop(paste("The number of starting values should be", nP))
  
  #TODO: we can probably support these, but don't at the moment
  if(mixture == "NB" || mixture == "ZIP")
    use_rgdual = FALSE
  # these we likely cannot support
  if(dynamics == "ricker" || dynamics == "gompertz")
    use_rgdual = FALSE
  
  ym <- matrix(y, nrow=M)
  
  # if(use_rgdual == FALSE) {
    # K is the support truncation parameter for abundance
    if(missing(K)) {
      # if not provided, it is set to a constant value
      #TODO: we've found that using the following scheme for setting K is significantly more robust:
      # Y <- max(y, na.rm=T)
      # scale <- 0.3856 # (determined empirically)
      # K <- ceiling((scale * Y + 10) / p)
      K <- max(y, na.rm=T) + 20
      #warning("K was not specified and was set to ", K, ".")
    }
    if(K <= max(y, na.rm = TRUE))
      stop("specified K is too small. Try a value larger than any observation")
    k  <- 0:K
    lk <- length(k)
    
    #TODO: describe these indices...
    # Create indices (should be written in C++)
    I <- cbind(rep(k, times=lk),
               rep(k, each=lk))
    I1 <- I[I[,1] <= I[,2],]
    Ib <- Ip <- list()
    for(i in 1:nrow(I)) {
      Z <- 0:min(I[i,])
      Ib[[i]] <- which((I1[,1] %in% Z) & (I1[,2]==I[i,1])) - 1
      Ip[[i]] <- as.integer(I[i,2]-Z)
    }
  # }
  
  # the previous pcount_open objective function, for when rgdual is unsupported
  nll_um <- function(parms, debug=FALSE) {
    beta.lam <- parms[1:nAP]
    beta.gam <- parms[(nAP+1):(nAP+nGP)]
    beta.om <- parms[(nAP+nGP+1):(nAP+nGP+nOP)]
    beta.p <- parms[(nAP+nGP+nOP+1):(nAP+nGP+nOP+nDP)]
    beta.iota <- parms[(nAP+nGP+nOP+nDP+1):(nAP+nGP+nOP+nDP+nIP)]
    log.alpha <- 1
    if(mixture %in% c("NB", "ZIP"))
      log.alpha <- parms[nP]
    if(debug) browser()
    .Call("nll_pcountOpen",
          ym,
          Xlam, Xgam, Xom, Xp, Xiota,
          beta.lam, beta.gam, beta.om, beta.p, beta.iota, log.alpha,
          Xlam.offset, Xgam.offset, Xom.offset, Xp.offset, Xiota.offset,
          ytna, yna,
          lk, mixture, first, last, M, J, T,
          delta, dynamics, fix, go.dims, immigration,
          I, I1, Ib, Ip,
          PACKAGE = "unmarked")
  }
  
  # the objective function based on rgdual
  nll_rgd <- function(parms, debug=FALSE) {
    # parameter extraction (following model from nll_pcountOpen.cpp, which does this in C)
    
    # lambda = initial arrival
    if(nAP > 0) {
      beta.lam <- parms[1:nAP]
      
      # transform lambda
      lambda <- as.vector(exp(Xlam * beta.lam + Xlam.offset))
    }
    # omega = survivorship rate
    if(nOP > 0) {
      beta.om <- parms[(nAP+nGP+1):(nAP+nGP+nOP)]
      
      # transform omega
      if(fix != "omega" && dynamics != "trend") {
        if(dynamics == "ricker" || dynamics == "gompertz") {
          omega <- as.vector(exp(Xom * beta.om + Xom.offset))
        } else {
          omega <- as.vector(logistic(Xom * beta.om + Xom.offset))
        }
      }
    }
    # gamma = recruitment rate
    if(nGP > 0) {
      beta.gam <- parms[(nAP+1):(nAP+nGP)]
      
      # transform gamma
      if(dynamics == "notrend") {
        #TODO: notrend support
        # gamma <- (1-om) % lamMat
      } else {
        if(fix != "gamma") {
          gamma <- as.vector(exp(Xgam * beta.gam + Xgam.offset))
        }
      }
    }
    # p = detection rate (also called det)
    if(nDP > 0) {
      beta.p <- parms[(nAP+nGP+nOP+1):(nAP+nGP+nOP+nDP)]
      
      # transform p
      det <- as.vector(logistic(Xp * beta.p + Xp.offset))
    }
    # iota = immigration rate
    if(nIP > 0) {
      beta.iota <- parms[(nAP+nGP+nOP+nDP+1):(nAP+nGP+nOP+nDP+nIP)]
      
      # transform iota
      iota <- as.vector(exp(Xiota * beta.iota + Xiota.offset))
    }
    # log.alpha = dispersion
    log.alpha <- 1
    if(mixture %in% c("NB", "ZIP")) {
      log.alpha <- parms[nP]
      
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
        return(pgf.bernoulli(s, theta) * pgf.poisson(s, theta))
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
      ym_double <- as.double(ym[i.site,])
      Alpha <- forward(ym_double, arrival.pgf, theta.arrival, offspring.pgf, theta.offspring, theta.observ, d=0)
      if(debug) browser()
      
      nll <- nll - forward.ll(Alpha)
    }
    
    return(nll)
  }
  
  if(missing(starts))
    starts <- rep(0, nP)
  
  fm <- optim(starts, nll_rgd, method=method, hessian=se, ...)
  # fm2 <- optim(starts, nll_um, method=method, hessian=se, ...)
  
  opt <- fm
  ests <- fm$par
  if(identical(mixture,"NB"))
    nbParm <- "alpha"
  else if(identical(mixture, "ZIP"))
    nbParm <- "psi"
  else
    nbParm <- character(0)
  names(ests) <- c(lamParms, gamParms, omParms, detParms, iotaParms, nbParm)
  if(se) {
    covMat <- tryCatch(solve(fm$hessian), error=function(x)
      simpleError("Hessian is not invertible. Try using fewer covariates or providing starting values."))
    if(class(covMat)[1] == "simpleError") {
      print(covMat$message)
      covMat <- matrix(NA, nP, nP)
    }
  } else covMat <- matrix(NA, nP, nP)
  
  fmAIC <- 2*fm$value + 2*nP
  
  lamEstimates <- unmarkedEstimate(name = "Abundance", short.name = "lam",
                                   estimates = ests[1:nAP], covMat = as.matrix(covMat[1:nAP,1:nAP]),
                                   invlink = "exp", invlinkGrad = "exp")
  detEstimates <- unmarkedEstimate(name = "Detection", short.name = "p",
                                   estimates = ests[(nAP+nGP+nOP+1) : (nAP+nGP+nOP+nDP)],
                                   covMat = as.matrix(covMat[(nAP+nGP+nOP+1) : (nAP+nGP+nOP+nDP),
                                                             (nAP+nGP+nOP+1) : (nAP+nGP+nOP+nDP)]),
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
                                                     estimates = ests[(nAP+1) : (nAP+nGP)], covMat = as.matrix(covMat[(nAP+1) :
                                                                                                                        (nAP+nGP), (nAP+1) : (nAP+nGP)]),
                                                     invlink = "exp", invlinkGrad = "exp")
  if(!(identical(fix, "omega") | identical(dynamics, "trend"))) {
    if(identical(dynamics, "constant") | identical(dynamics, "autoreg") | identical(dynamics, "notrend"))
      estimateList@estimates$omega <- unmarkedEstimate(
        name="Apparent Survival",
        short.name = "omega", estimates = ests[(nAP+nGP+1) :(nAP+nGP+nOP)],
        covMat = as.matrix(covMat[(nAP+nGP+1) : (nAP+nGP+nOP),
                                  (nAP+nGP+1) : (nAP+nGP+nOP)]),
        invlink = "logistic", invlinkGrad = "logistic.grad")
    else if(identical(dynamics, "ricker"))
      estimateList@estimates$omega <- unmarkedEstimate(
        name="Carrying Capacity",
        short.name = "omCarCap", estimates = ests[(nAP+nGP+1) :(nAP+nGP+nOP)],
        covMat = as.matrix(covMat[(nAP+nGP+1) : (nAP+nGP+nOP),
                                  (nAP+nGP+1) : (nAP+nGP+nOP)]),
        invlink = "exp", invlinkGrad = "exp")
    else
      estimateList@estimates$omega <- unmarkedEstimate(
        name="Carrying Capacity",
        short.name = "omCarCap", estimates = ests[(nAP+nGP+1) :(nAP+nGP+nOP)],
        covMat = as.matrix(covMat[(nAP+nGP+1) : (nAP+nGP+nOP),
                                  (nAP+nGP+1) : (nAP+nGP+nOP)]),
        invlink = "exp", invlinkGrad = "exp")
  }
  estimateList@estimates$det <- detEstimates
  if(immigration) {
    estimateList@estimates$iota <- unmarkedEstimate(
      name="Immigration",
      short.name = "iota", estimates = ests[(nAP+nGP+nOP+nDP+1) :(nAP+nGP+nOP+nDP+nIP)],
      covMat = as.matrix(covMat[(nAP+nGP+nOP+nDP+1) : (nAP+nGP+nOP+nDP+nIP),
                                (nAP+nGP+nOP+nDP+1) : (nAP+nGP+nOP+nDP+nIP)]),
      invlink = "exp", invlinkGrad = "exp")
  }
  if(identical(mixture, "NB")) {
    estimateList@estimates$alpha <- unmarkedEstimate(name = "Dispersion",
                                                     short.name = "alpha", estimates = ests[nP],
                                                     covMat = as.matrix(covMat[nP, nP]), invlink = "exp",
                                                     invlinkGrad = "exp")
  }
  if(identical(mixture, "ZIP")) {
    estimateList@estimates$psi <- unmarkedEstimate(name = "Zero-inflation",
                                                   short.name = "psi", estimates = ests[nP],
                                                   covMat = as.matrix(covMat[nP, nP]), invlink = "logistic",
                                                   invlinkGrad = "logistic.grad")
  }
  umfit <- new("unmarkedFitPCO", fitType = "pcountOpen",
               call = match.call(), formula = formula, formlist = formlist, data = data,
               sitesRemoved=D$removed.sites, estimates = estimateList, AIC = fmAIC,
               opt = opt, negLogLike = fm$value, nllFun = nll_rgd, K = K, mixture = mixture,
               dynamics = dynamics, immigration = immigration)
  return(umfit)
}

environment(pcountOpen_rgdual) <- asNamespace('unmarked')





