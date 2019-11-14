source("apgffwd.R")

library(unmarked)

# todo list before release:
# unsupported models = {ricker, gompertz, notrend}
#   ricker/gompertz probably aren't, notrend unsure
# unsupported mixtures = {NB, ZIP}, probably just adding a PGF selection
# add info to the return data structure indicating which likelihood function was selected
# support for repeated counts
# implement the PGFFA symbolic algorithm for Poisson HMM
# implement gdual (the non-ls version) in case it's faster?

pcountOpen_apgffwd <- function(lambdaformula, gammaformula, omegaformula, pformula,
                              data, 
                              mixture     = 'P', 
                              K, 
                              dynamics    = 'constant',
                              fix         = 'none',
                              starts      = 'auto', 
                              method      = 'L-BFGS-B', 
                              se          = TRUE, 
                              immigration = FALSE, 
                              iotaformula = ~1,
                              eval.at     = NULL,
                              nll.fun     = 'apgffwd',
                              n.attempts  = 5,
                              rho.prior.mode = 0.5,
                              rho.prior.strength = NA,
                              fix.lambda = FALSE,
                              fix.gamma  = FALSE,
                              fix.omega  = FALSE,
                              fix.rho    = FALSE,
                              fix.iota   = FALSE,
                              transform  = 'auto',
                              maxit      = 'default',
                              ...) {
  mixture  <- match.arg(mixture,  c('P', 'NB', 'ZIP'))
  dynamics <- match.arg(dynamics, c('constant', 'autoreg', 'notrend', 'trend', 'ricker', 'gompertz', 'trend-NB'))
  fix      <- match.arg(fix,      c('none', 'gamma', 'omega'))
  nll.fun  <- match.arg(nll.fun,  c('auto', 'apgffwd', 'trunc'))
  # starts   <- match.arg(starts,   c('auto', NA, numeric, D$nP))
  
  #TODO: unused error handling from unmarked, safe to remove?
  ## if(identical(dynamics, "notrend") &
  ##    !identical(lambdaformula, omegaformula))
  ##     stop("lambdaformula and omegaformula must be identical for notrend model")
  
  # browser()
  
  if((identical(dynamics, "constant") | identical(dynamics, "notrend")) & immigration)
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
  can.rgdual = TRUE
  # if(J > 1)
  #   use_rgdual = FALSE
  
  #TODO: we can probably support these, but don't at the moment
  if(mixture == "ZIP")
    can.rgdual = FALSE
  # these we likely cannot support
  if(dynamics == "ricker" || dynamics == "gompertz")
    can.rgdual = FALSE
  if(mixture != "P" && (dynamics %in% c("constant", "notrend")))
    can.rgdual = FALSE
  
  if(identical(nll.fun, 'apgffwd') && !can.rgdual)
    stop("Unable to evaluate model using rgdual.")
  
  # # if possible, use rgdual unless trunc specified
  # can.rgdual <- use_rgdual && !identical(nll.fun, 'trunc')
  
  # update the design matrix to handle missing data/missing parmeters
  D <- pcountOpen_handle_missing(D, M, T, J)
  
  # process the parameters specified in the design matrix
  D <- pcountOpen_parameter_structure(D, dynamics, immigration, mixture, fix)
  
  # fix some parameters
  if(fix.lambda != FALSE)
    nAP <- 0
  else
    nAP <- D$nAP
  if(fix.gamma != FALSE)
    nGP <- 0
  else
    nGP <- D$nGP
  if(fix.omega != FALSE)
    nOP <- 0
  else
    nOP <- D$nOP
  if(fix.rho != FALSE)
    nDP <- 0
  else
    nDP <- D$nDP
  if(fix.iota != FALSE)
    nIP <- 0
  else
    nIP <- D$nIP
  nP <- nAP + nGP + nOP + nDP + nIP + (mixture != 'P')
  
  if(!missing(starts) && is.numeric(starts) && length(starts) > 1 && length(starts) != D$nP)
    stop(paste("Starting values provided but could not be matched to the number of parameters: ", D$nP))
  
  D$ym <- matrix(D$y, nrow=M)

  # resolve auto in one of the nll methods
  if(identical(nll.fun, "auto")) {
    if(can.rgdual == FALSE)
      nll.fun <- "trunc"
    else
      nll.fun <- "apgffwd"
  }

  if(identical(nll.fun, "apgffwd") && !identical(method, "L-BFGS-B")) {
    warning("pcountOpen_apgffwd uses weak parameter bounds and L-BFGS-B for optimization, other method choices are for testing/evaluation only.")
  }
  
  rho.prior.alpha <- rho.prior.mode * rho.prior.strength + 1
  rho.prior.beta  <- (1 - rho.prior.mode) * rho.prior.strength + 1
  
  # set some parameters only needed by the truncated method
  if(identical(nll.fun, "trunc")) {
    # K is the support truncation parameter for abundance
    if(missing(K))
      K <- K.default(D$ym)
    
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
  
  if(identical(transform, 'auto')) {
    if(identical(nll.fun, 'apgffwd')) {
      if(identical(method, 'L-BFGS-B'))
        transform <- FALSE
      else
        transform <- TRUE
    } else
      transform <- TRUE
  }
  
  # invert the transformation for any fixed parameters
  if(transform) {
    if(fix.lambda != FALSE & fix.lambda != "auto")
      fix.lambda <- log(fix.lambda)
    if(fix.gamma != FALSE)
      fix.gamma <- log(fix.gamma)
    if(fix.omega != FALSE) {
      if(identical(dynamics, "ricker") || identical(dynamics, "gompertz"))
        fix.omega <- log(fix.omega)
      else
        fix.omega <- qlogis(fix.omega) # qlogis == logit
    }
    if(fix.rho != FALSE)
      fix.rho <- qlogis(fix.rho)
    if(fix.iota != FALSE)
      fix.iota <- log(fix.iota)
  }
  
  if(identical(nll.fun, "trunc")) {
    # the previous pcountOpen objective function, for when rgdual is unsupported
    objective.fun <- function(parms, debug=FALSE) {
      if(fix.lambda != FALSE)
        beta.lam  <- fix.lambda
      else
        beta.lam  <- parms[1:nAP]
      if(fix.gamma != FALSE)
        beta.gam  <- fix.gamma
      else
        beta.gam  <- parms[(nAP+1):(nAP+nGP)]
      if(fix.omega != FALSE)
        beta.om   <- fix.omega
      else
        beta.om   <- parms[(nAP+nGP+1):(nAP+nGP+nOP)]
      if(fix.rho != FALSE)
        beta.p    <- fix.rho
      else
        beta.p    <- parms[(nAP+nGP+nOP+1):(nAP+nGP+nOP+nDP)]
      if(fix.iota != FALSE)
        beta.iota <- fix.iota
      else
        beta.iota <- parms[(nAP+nGP+nOP+nDP+1):(nAP+nGP+nOP+nDP+nIP)]
      log.alpha <- 1
      # browser()
      if(mixture %in% c("NB", "ZIP"))
        log.alpha <- parms[nP]
      
      # browser()
      
      if(beta.lam == 'auto') {
        beta.lam <- 1
        
        # browser()
        first.nonmissingvals <- apply(D$ym, 1, function(row) { row[min(which(!is.na(row)))] } )
        if(!transform) {
          if(beta.p == 0) {
            Xlam.offset <- first.nonmissingvals
          } else {
            Xlam.offset <- (first.nonmissingvals / beta.p)
          }
        } else {
          if(beta.p == -Inf) {
            Xlam.offset <- log(first.nonmissingvals)
          } else {
            Xlam.offset <- log(first.nonmissingvals / plogis(beta.p))
          }
        }
      } else {
        Xlam.offset <- D$Xlam.offset
      }
      
      # browser()
      nll <- .Call("nll_pcountOpen",
            D$ym,
            D$Xlam, D$Xgam, D$Xom, D$Xp, D$Xiota,
            beta.lam, beta.gam, beta.om, beta.p, beta.iota, log.alpha,
            Xlam.offset, D$Xgam.offset, D$Xom.offset, D$Xp.offset, D$Xiota.offset,
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
      
      # if(!is.na(rho.prior.strength)) {
      #   if(transform)
      #     nll <- nll - dbeta(logistic(beta.p[1]), rho.prior.alpha, rho.prior.beta, log = TRUE)
      #   else
      #     nll <- nll - dbeta(beta.p[1], rho.prior.alpha, rho.prior.beta, log = TRUE)
      # }
      return(nll)
    }
  } else { # use_rgdual == TRUE
    # the objective function based on rgdual
    objective.fun <- function(parms, debug=FALSE) {
      # parameter extraction (following model from nll_pcountOpen.cpp, which does a lot of this in C)
      beta.lam  <- parms[1                          : nAP]
      beta.gam  <- parms[(nAP+1)                  :(nAP+nGP)]
      beta.om   <- parms[(nAP+nGP+1)            :(nAP+nGP+nOP)]
      beta.p    <- parms[(nAP+nGP+nOP+1)      :(nAP+nGP+nOP+nDP)]
      beta.iota <- parms[(nAP+nGP+nOP+nDP+1):(nAP+nGP+nOP+nDP+nIP)]
      
      # lambda = initial arrival
      if(nAP > 0) {
        if(transform == TRUE) {
          lambda <- as.vector(exp(D$Xlam * beta.lam + D$Xlam.offset))
        } else {
          lambda <- as.vector(D$Xlam * beta.lam + D$Xlam.offset)
        }
      } else {
        if(identical(fix.lambda, 'auto'))
          lambda <- NaN
        else 
          lambda <- as.vector(fix.lambda)
      }

      # omega = survivorship rate
      if(nOP > 0) {
        if(fix != "omega") {
          if(transform == TRUE) {
            if(dynamics == "ricker" || dynamics == "gompertz") {
              omega <- as.vector(exp(D$Xom * beta.om + D$Xom.offset))
            } else {
              omega <- as.vector(logistic(D$Xom * beta.om + D$Xom.offset))
            }
          } else {
            omega <- as.vector(D$Xom * beta.om + D$Xom.offset)
          }
        }
      } else {
        omega <- as.vector(fix.omega)
      }

      # gamma = recruitment rate
      if(nGP > 0) {
        #TODO: notrend may transform differently?
        if(fix != "gamma") {
          if(transform == TRUE) {
            gamma <- as.vector(exp(D$Xgam * beta.gam + D$Xgam.offset))
          } else {
            gamma <- as.vector(D$Xgam * beta.gam + D$Xgam.offset)
          }
        }
      } else {
        gamma <- as.vector(fix.gamma)
      }

      # det = detection rate (also called p)
      if(nDP > 0) {
        if(transform == TRUE) {
          det <- as.vector(logistic(D$Xp * beta.p + D$Xp.offset))
        } else {
          det <- as.vector(D$Xp * beta.p + D$Xp.offset)
        }
      } else {
        det <- as.vector(fix.rho)
      }

      # iota = immigration rate
      if(nIP > 0) {
        if(transform == TRUE) {
          iota <- as.vector(exp(D$Xiota * beta.iota + D$Xiota.offset))
        } else {
          iota <- as.vector(D$Xiota * beta.iota + D$Xiota.offset)
        }
      } else {
        iota <- as.vector(fix.iota)
      }

      # alpha = dispersion, will be the last parameter if mixture uses it
      if(mixture == "NB" || mixture == "ZIP") {
        alpha <- parms[nP]
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
      if(identical(mixture, "P")) {
        if(identical(dynamics,"constant")) {
          # arrivals are always Poisson, initial arrivals are parameterized by lambda, then by gamma
          arrival.pgf   <- pgf.poisson
          theta.arrival <- data.frame(lambda = c(lambda[1], array(gamma, T - 1)))
        } else if(identical(dynamics, "notrend")) {
          arrival.pgf   <- pgf.poisson
          theta.arrival <- data.frame(lambda = c(lambda[1], array((1 - omega[1])*lambda[1], T - 1)))
        } else if(immigration) {
          # arrivals are always Poisson, initial arrivals are parameterized by lambda, then by gamma
          arrival.pgf   <- pgf.poisson
          theta.arrival <- data.frame(lambda = c(lambda[1], array(iota, T - 1)))
        } else {
          # initial arrivals are Poisson w/ mean lambda, then zero always
          arrival.pgf   <- pgf.poisson
          theta.arrival <- data.frame(lambda = c(lambda[1], array(0, T - 1)))
        }
      } else if(identical(mixture, "NB")) {
        arrival.pgf   <- pgf.negbin
        if(immigration) {
          # arrivals are always Poisson, initial arrivals are parameterized by lambda, then by gamma
          theta.arrival <- data.frame(r = c(lambda[1], array(iota, T - 1)),
                                      p = array(alpha, T))
        } else {
          # initial arrivals are Poisson w/ mean lambda, then zero always
          theta.arrival <- data.frame(r = c(lambda[1], array(1, T - 1)),
                                      p = c(alpha, array(1, T - 1)))
        }
      }
      
      # select the offspring distribution
      if(dynamics %in% c("constant", "notrend")) {
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
      } else if (dynamics %in% c("trend-NB")) {
        offspring.pgf   <- pgf.geometric
        theta.offspring <- data.frame(p = array(gamma, T - 1))
      }
      #TODO: all mixtures
      
      # detection dist'n is always binomial
      theta.observ    <- data.frame(p=array(det, T))
      
      nll <- 0
      tryCatch({
        for (i.site in 1:M) {
          i.nonmissing <- which(!is.na(D$ym[i.site,]))
          
          # at the moment our C wrappers are not overloaded for integers, so everything needs to be doubles
          # TODO: update C wrappers to take integers (as is, this cannot be done w/ automatic dispatch)
          ym_double <- as.double(D$ym[i.site,min(i.nonmissing):max(i.nonmissing)])
          # print('doing a try')
          
          # print(ym_double[1])
          
          if(identical(fix.lambda, 'auto')) {
            if(identical(mixture, "P")) {
              lambda.auto <- ym_double[1] / theta.observ[1,]
              theta.arrival[1,'lambda'] <- lambda.auto
              if(identical(dynamics, 'notrend')) {
                theta.arrival[2:nrow(theta.arrival),'lambda'] <- (1 - omega[1]) * lambda.auto
              }
            } else if(identical(mixture, "NB")) {
              lambda.auto <- ym_double[1] / theta.observ[1,] * theta.arrival[1,'p'] / (1 - theta.arrival[1,'p'])
              theta.arrival[1,'r'] <- lambda.auto
              if(identical(dynamics, 'notrend')) {
                stop()
                # theta.arrival[2:nrow(theta.arrival),'r'] <- (1 - omega) * lambda.auto
              }
            }
          }
          
          # browser()
          
          nll.iter <- apgffwd(ym_double, arrival.pgf, theta.arrival, offspring.pgf, theta.offspring, theta.observ, d=0)
          
          # nll is currently a ls (log-magnitude and sign) number
          # if it is a negative number (and not essentially zero), something went wrong
          stopifnot(nll.iter$sign >= 0 || nll.iter$mag < 1e-6)
          nll <- nll - nll.iter$mag
        }
        
        if(!is.na(rho.prior.strength)) {
          if(transform)
            nll <- nll - dbeta(logistic(beta.p[1]), rho.prior.alpha, rho.prior.beta, log = TRUE)
          nll <- nll - dbeta(beta.p[1], rho.prior.alpha, rho.prior.beta, log = TRUE)
        }
        
        # browser()
        
        # if(det[1] > 0.99)
        #   browser()
        
        if(!is.finite(nll))
          browser()
      }, error = function(e) {
        warning("Something went wrong in apgffwd. Returning NLL == Inf")
        # browser()
        return(Inf)
      })
      
      return(nll)
    }
  } # /if use_rgdual == FALSE
  
  # if eval.at provided and consistent, just call the objective function once and return
  if(!missing(eval.at) && !is.null(eval.at) && length(eval.at) == nP)
    return(objective.fun(eval.at))
  
  if(method == "L-BFGS-B") {
    # by default, constrain parameters weakly to positive reals w/ extreme upper bound
    lower = rep(1e-3, nP)
    upper = rep(1e6,  nP)
    
    if(nGP > 0 && dynamics %in% c("trend-NB")) {
      lower[(nAP+1):(nAP+nGP)] <- 1e-3
      upper[(nAP+1):(nAP+nGP)] <- 1 - (1e-3)
    }
    
    if(nOP > 0) {
      lower[(nAP+nGP+1):(nAP+nGP+nOP)] <- 1e-3
      upper[(nAP+nGP+1):(nAP+nGP+nOP)] <- 1 - (1e-3)
    }

    # detection param in 0 to 1
    if(nDP > 0) {
      lower[(nAP+nGP+nOP+1):(nAP+nGP+nOP+nDP)] <- 1e-3
      upper[(nAP+nGP+nOP+1):(nAP+nGP+nOP+nDP)] <- 1 - (1e-3)
    }
    
    if(mixture %in% c("NB", "ZIP")) {
      lower[(nAP+nGP+nOP+1):(nAP+nGP+nOP+nDP)] <- 1e-3
      upper[(nAP+nGP+nOP+1):(nAP+nGP+nOP+nDP)] <- 1 - (1e-3)
    }
  }

  # note: new default start option that tries to be a little smarter
  if(is.na(starts))
    starts <- rep(0, nP)
  else if(is.numeric(starts) && length(starts) == 1)
    starts <- rep(starts, nP)
  else if(identical(starts, "auto")) {
    # if(identical(method, "L-BFGS-B")) {
    # automatically choose "reasonable" starting params
    # generally 0.5 for [0,1] params and some function of y for unconstrained params
    starts <- rep(0, nP)
    if(nAP > 0) {
      if(transform)
        starts[1:nAP] <- log(mean(D$ym[,1], na.rm=TRUE))
      else
        starts[1:nAP] <- mean(D$ym[,1], na.rm=TRUE)
    }
    if(nGP > 0) {
      if(dynamics %in% c("trend-NB")) {
        starts[(nAP+1):(nAP+nGP)] <- 0.5
      } else {
        if(transform)
          starts[(nAP+1):(nAP+nGP)] <- log(max(mean(D$ym[,2:ncol(D$ym)] - D$ym[,1:(ncol(D$ym) - 1)], na.rm = TRUE), 0.1, na.rm = TRUE))
        else
          starts[(nAP+1):(nAP+nGP)] <- max(mean(D$ym[,2:ncol(D$ym)] - D$ym[,1:(ncol(D$ym) - 1)], na.rm = TRUE), 0.1, na.rm = TRUE)
      }
    }
    if(nOP > 0) {
      if(identical(dynamics, "ricker") || identical(dynamics, "gompertz")) {
        if(transform)
          starts[(nAP+nGP+1):(nAP+nGP+nOP)] <- log(1)
        else
          starts[(nAP+nGP+1):(nAP+nGP+nOP)] <- 1
      } else {
        if(transform)
          starts[(nAP+nGP+1):(nAP+nGP+nOP)] <- qlogis(0.5) #qlogis = logit
        else
          starts[(nAP+nGP+1):(nAP+nGP+nOP)] <- 0.5
      }
    }
    if(nDP > 0) {
      if(transform)
        starts[(nAP+nGP+nOP+1):(nAP+nGP+nOP+nDP)] <- qlogis(0.5)
      else
        starts[(nAP+nGP+nOP+1):(nAP+nGP+nOP+nDP)] <- 0.5
    }
    if(nIP > 0) {
      if(transform)
        starts[(nAP+nGP+nOP+nDP+1):(nAP+nGP+nOP+nDP+nIP)] <- log(max(mean(D$ym[,2:ncol(D$ym)] - D$ym[,1:(ncol(D$ym) - 1)], na.rm=TRUE), 0.1, na.rm=TRUE))
      else
        starts[(nAP+nGP+nOP+nDP+1):(nAP+nGP+nOP+nDP+nIP)] <- max(mean(D$ym[,2:ncol(D$ym)] - D$ym[,1:(ncol(D$ym) - 1)], na.rm=TRUE), 0.1, na.rm=TRUE)
    }
    if(mixture %in% c("NB", "ZIP")) {
      starts[nP] <- 0.5
    }
    # browser()
    # } else {
    #   starts <- rep(0, nP)
    # }
  }
  
  # browser()
  # if (nAP > 0)
  #   starts[1:nAP] <- log(max(D$y[,1]))
  
  # browser()
  
  control = list()
  if(identical(method, 'L-BFGS-B')) {
    control$trace <- 1
    if(!identical(maxit, 'default'))
      control$maxit <- maxit
  } else {
    control$trace <- FALSE
    if(!identical(maxit, 'default'))
      control$maxit <- maxit
  }
  
  i.attempt <- 1
  fm <- NULL
  
  # browser()
  while(i.attempt <= n.attempts && is.null(fm)) {
    # tryCatch({
      # fm <- optim(starts, objective.fun, method=method, hessian=se, ...)
      if(method == "L-BFGS-B")
        fm <- optim(starts, objective.fun, method=method, hessian=se, control = control, lower = lower, upper = upper, ...)
      else
        fm <- optim(starts, objective.fun, method=method, hessian=se, control = control, ...)
    # }, error = function(e) {
    #   browser()
    #   print(paste0('Attempt #', i.attempt, ' failed: ', e, '\nAdding noise to initial values.'))
    #   fm <- NULL
    #   starts <- starts + rexp(nP, 1e4)
    #   i.attempt <- i.attempt + 1
    # })
  }
  
  # browser()
  # if(fm$convergence > 49)
  #   browser()
  
  stopifnot(!is.null(fm))
  # fm <- optim(starts, objective.fun, method=method, hessian=se, ...)
  # fm2 <- optim(starts, nll_um, method=method, hessian=se, ...)
  
  opt <- fm
  
  # construct the vector of parameter estimates
  ests <- vector(D$nP, mode = 'numeric')
  # ests <- fm$par
  if(fix.lambda != FALSE) {
    if(identical(fix.lambda, 'auto')) {
      ests[1:D$nAP] <- NA
    } else {
      ests[1:D$nAP] <- fix.lambda
    }
  } else
    ests[1:D$nAP] <- fm$par[1:nAP]
  if(fix.gamma != FALSE)
    ests[(1+D$nAP):(D$nAP+D$nGP)] <- fix.gamma
  else
    ests[(1+D$nAP):(D$nAP+D$nGP)] <- fm$par[(1+nAP):(nAP+nGP)]
  if(fix.omega != FALSE)
    ests[(1+D$nAP+D$nGP):(D$nAP+D$nGP+D$nOP)] <- fix.omega
  else
    ests[(1+D$nAP+D$nGP):(D$nAP+D$nGP+D$nOP)] <- fm$par[(1+nAP+nGP):(nAP+nGP+nOP)]
  if(fix.rho != FALSE)
    ests[(1+D$nAP+D$nGP+D$nOP):(D$nAP+D$nGP+D$nOP+D$nDP)] <- fix.rho
  else
    ests[(1+D$nAP+D$nGP+D$nOP):(D$nAP+D$nGP+D$nOP+D$nDP)] <- fm$par[(1+nAP+nGP+nOP):(nAP+nGP+nOP+nDP)]
  if(fix.iota != FALSE)
    ests[(1+D$nAP+D$nGP+D$nOP+D$nDP):(D$nAP+D$nGP+D$nOP+D$nDP+D$nIP)] <- fix.iota
  else
    ests[(1+D$nAP+D$nGP+D$nOP+D$nDP):(D$nAP+D$nGP+D$nOP+D$nDP+D$nIP)] <- fm$par[(1+nAP+nGP+nOP+nDP):(nAP+nGP+nOP+nDP+nIP)]
  if(mixture %in% c("NB", "ZIP"))
    ests[D$nP] <- fm$par[nP]
  
  # browser()
  
  if(identical(mixture,"NB"))
    nbParm <- "alpha"
  else if(identical(mixture, "ZIP"))
    nbParm <- "psi"
  else
    nbParm <- character(0)
  names(ests) <- c(D$lamParms, D$gamParms, D$omParms, D$detParms, D$iotaParms, D$alphaParms)
  if(se) {
    covMat <- tryCatch(solve(fm$hessian), error=function(x)
      simpleError("Hessian is not invertible. Try using fewer covariates or providing starting values."))
    if(class(covMat)[1] == "simpleError") {
      print(covMat$message)
      covMat <- matrix(NA, nP, nP)
    }
  } else covMat <- matrix(NA, nP, nP)
  
  # browser()
  
  fmAIC <- 2*fm$value + 2*nP
  
  if(fix.lambda == FALSE)
    lamEstimates <- unmarkedEstimate(name = "Abundance", short.name = "lam",
                                     estimates = ests[1:D$nAP], covMat = as.matrix(covMat[1:nAP,1:nAP]),
                                     invlink = "exp", invlinkGrad = "exp")
  else
    lamEstimates <- unmarkedEstimate(name = "Abundance", short.name = "lam",
                                     estimates = ests[1:D$nAP], covMat = matrix(,nrow=D$nAP,ncol=D$nAP),
                                     invlink = "exp", invlinkGrad = "exp")
  if(fix.rho == FALSE)
    detEstimates <- unmarkedEstimate(name = "Detection", short.name = "p",
                                     estimates = ests[(D$nAP+D$nGP+D$nOP+1) : (D$nAP+D$nGP+D$nOP+D$nDP)],
                                     covMat = as.matrix(covMat[(nAP+nGP+nOP+1) : (nAP+nGP+nOP+nDP),
                                                               (nAP+nGP+nOP+1) : (nAP+nGP+nOP+nDP)]),
                                     invlink = "logistic", invlinkGrad = "logistic.grad")
  else
    detEstimates <- unmarkedEstimate(name = "Detection", short.name = "p",
                                     estimates = ests[(D$nAP+D$nGP+D$nOP+1) : (D$nAP+D$nGP+D$nOP+D$nDP)],
                                     covMat = matrix(,nrow=D$nDP,ncol=D$nDP),
                                     invlink = "logistic", invlinkGrad = "logistic.grad")
  estimateList <- unmarkedEstimateList(list(lambda=lamEstimates))
  gamName <- switch(dynamics,
                    constant = "gamConst",
                    autoreg = "gamAR",
                    notrend = "",
                    trend = "gamTrend",
                    ricker = "gamRicker",
                    gompertz = "gamGomp",
                    "trend-NB" = "gamNBp")
  if(!(identical(fix, "gamma") | identical(dynamics, "notrend"))) {
    if(fix.gamma == FALSE)
      estimateList@estimates$gamma <- unmarkedEstimate(name =
                                                         ifelse(identical(dynamics, "constant") | identical(dynamics, "autoreg"),
                                                                "Recruitment", "Growth Rate"), short.name = gamName,
                                                       estimates = ests[(D$nAP+1) : (D$nAP+D$nGP)], covMat = as.matrix(covMat[(nAP+1) :
                                                                                                                          (nAP+nGP), (nAP+1) : (nAP+nGP)]),
                                                       invlink = "exp", invlinkGrad = "exp")
    else
      estimateList@estimates$gamma <- unmarkedEstimate(name =
                                                         ifelse(identical(dynamics, "constant") | identical(dynamics, "autoreg"),
                                                                "Recruitment", "Growth Rate"), short.name = gamName,
                                                       estimates = ests[(D$nAP+1) : (D$nAP+D$nGP)], covMat = matrix(,nrow=D$nGP,ncol=D$nGP),
                                                       invlink = "exp", invlinkGrad = "exp")
  }
  if(!(identical(fix, "omega") | (dynamics %in% c("trend", "trend-NB")))) {
    if(fix.omega == FALSE)
      om.cov <- as.matrix(covMat[(nAP+nGP+1) : (nAP+nGP+nOP),
                                 (nAP+nGP+1) : (nAP+nGP+nOP)])
    else
      om.cov <- matrix(,nrow=D$nOP,ncol=D$nOP)
    if(identical(dynamics, "constant") | identical(dynamics, "autoreg") | identical(dynamics, "notrend"))
      estimateList@estimates$omega <- unmarkedEstimate(
        name="Apparent Survival",
        short.name = "omega", estimates = ests[(D$nAP+D$nGP+1) :(D$nAP+D$nGP+D$nOP)],
        covMat = om.cov,
        invlink = "logistic", invlinkGrad = "logistic.grad")
    else if(identical(dynamics, "ricker"))
      estimateList@estimates$omega <- unmarkedEstimate(
        name="Carrying Capacity",
        short.name = "omCarCap", estimates = ests[(D$nAP+D$nGP+1) :(D$nAP+D$nGP+D$nOP)],
        covMat = om.cov,
        invlink = "exp", invlinkGrad = "exp")
    else
      estimateList@estimates$omega <- unmarkedEstimate(
        name="Carrying Capacity",
        short.name = "omCarCap", estimates = ests[(D$nAP+D$nGP+1) :(D$nAP+D$nGP+D$nOP)],
        covMat = om.cov,
        invlink = "exp", invlinkGrad = "exp")
  }
  estimateList@estimates$det <- detEstimates
  if(immigration) {
    if(fix.iota == FALSE)
      imm.cov <- as.matrix(covMat[(nAP+nGP+nOP+nDP+1) : (nAP+nGP+nOP+nDP+nIP),
                                  (nAP+nGP+nOP+nDP+1) : (nAP+nGP+nOP+nDP+nIP)])
    else
      imm.cov <- matrix(,nrow=D$nIP,ncol=D$nIP)
    estimateList@estimates$iota <- unmarkedEstimate(
      name="Immigration",
      short.name = "iota", estimates = ests[(D$nAP+D$nGP+D$nOP+D$nDP+1) :(D$nAP+D$nGP+D$nOP+D$nDP+D$nIP)],
      covMat = imm.cov,
      invlink = "exp", invlinkGrad = "exp")
  }
  if(identical(mixture, "NB")) {
    estimateList@estimates$alpha <- unmarkedEstimate(name = "Dispersion",
                                                     short.name = "alpha", estimates = ests[D$nP],
                                                     covMat = as.matrix(covMat[nP,nP]), invlink = "exp",
                                                     invlinkGrad = "exp")
  }
  if(identical(mixture, "ZIP")) {
    estimateList@estimates$psi <- unmarkedEstimate(name = "Zero-inflation",
                                                   short.name = "psi", estimates = ests[D$nP],
                                                   covMat = as.matrix(covMat[nP,nP]), invlink = "logistic",
                                                   invlinkGrad = "logistic.grad")
  }
  
  umfit <- new("unmarkedFitPCO", fitType = "pcountOpen",
               call = match.call(), formula = formula, formlist = formlist, data = data,
               sitesRemoved=D$removed.sites, estimates = estimateList, AIC = fmAIC,
               opt = opt, negLogLike = fm$value, nllFun = objective.fun, K = K, mixture = mixture,
               dynamics = dynamics, immigration = immigration)
  # browser()
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
  if(dynamics %in% c("trend", "trend-NB") && !identical(fix, 'omega')) {
    if(D$nOP > 1) stop("omega covariates not allowed when dynamics==trend")
    
    D$nOP     <- 0
    D$omParms <- character(0)
  }
  if(identical(dynamics, "notrend") && !identical(fix, 'gamma')) {
    if(D$nGP > 1) stop("gamma covariates not allowed when dynamics==notrend")
    
    D$nGP      <- 0
    D$gamParms <- character(0)
  }
  
  if(mixture %in% c("NB", "ZIP"))
    D$alphaParms <- "Point"
  
  # total number of parameters (+1 if mixture includes log.alpha)
  D$nP <- D$nAP + D$nGP + D$nOP + D$nDP + D$nIP + (mixture == 'NB' || mixture == 'ZIP')
  
  return(D)
}

# set a K which is consistent with the data
K.default <- function(y, p = NULL) {
  Y <- max(y, na.rm=T)
  
  if(!is.null(p) && is.finite(p)) {
    #we've found that using the following scheme for setting K is significantly more robust:
    # scale <- 0.3856 # (determined empirically)
    # K <- max(Y + 20, ceiling((scale * Y + 10) / p))
    K <- max(Y + 20, ceiling(Y / p))
  } else {
    # K <- max(ceiling(scale * Y + 10)
    K <- Y + 20
  }
  # print(paste0("K was not specified and was set to ", K, "."))
  return(K)
}

environment(pcountOpen_apgffwd) <- environment(pcountOpen)
environment(pcountOpen_handle_missing) <- environment(pcountOpen)
environment(pcountOpen_parameter_structure) <- environment(pcountOpen)
environment(K.default) <- environment(pcountOpen)

# environment(pcountOpen_rgdual) <- as.environment('namespace:unmarked')
# environment(pcountOpen_handle_missing) <- as.environment('package:unmarked')
# environment(pcountOpen_parameter_structure) <- as.environment('package:unmarked')
# environment(K.default) <- as.environment('package:unmarked')