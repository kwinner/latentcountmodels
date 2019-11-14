library(unmarked)
source('pcountOpen_rgdual.R')
# source('pcountOpen_apgffwd.R')
source('pcountOpen_constrained_apgffwd.R')

do.experiment = TRUE

if(do.experiment) {

do.pco     <- TRUE
do.rgdfwd  <- TRUE
do.apgffwd <- TRUE

pco_method     <- 'Nelder-Mead'
apgffwd.method <- 'L-BFGS-B'

fix.lambda = FALSE
fix.gamma  = FALSE
fix.omega  = FALSE
fix.rho    = TRUE
fix.iota   = FALSE

dynamics    <- "trend"
immigration <- TRUE
M <- 4
T <- 8
lambda <- 20
iota <- 4
gamma <- 0.95
omega <- 0.25
p <- 0.5

# rho.prior.strength <- 100
# test_vals <- seq(15, 75, 30)
# test_vals <- c(15)
test_vals <- seq(0.05, 0.95, 0.1)
# test_vals <- c(0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10)
# test_vals <- 0.5
n.experiments <- length(test_vals)
n.reps <- 10

# runtime.record <- data.frame()
runtime_unmarked <- matrix(NA, n.experiments, n.reps)
runtime_rgdfwd   <- matrix(NA, n.experiments, n.reps)
runtime_apgffwd  <- matrix(NA, n.experiments, n.reps)

lambda_record_gen <- vector(n.experiments, mode='numeric')
iota_record_gen   <- vector(n.experiments, mode='numeric')
gamma_record_gen  <- vector(n.experiments, mode='numeric')
omega_record_gen  <- vector(n.experiments, mode='numeric')
p_record_gen      <- vector(n.experiments, mode='numeric')

y_record          <- vector(n.experiments * n.reps, mode='list')
K_record          <- matrix(NA, n.experiments, n.reps)

lambda_record_pco <- matrix(NA, n.experiments, n.reps)
gamma_record_pco <- matrix(NA, n.experiments, n.reps)
omega_record_pco <- matrix(NA, n.experiments, n.reps)
det_record_pco <- matrix(NA, n.experiments, n.reps)
iota_record_pco <- matrix(NA, n.experiments, n.reps)

lambda_record_rgdual <- matrix(NA, n.experiments, n.reps)
gamma_record_rgdual <- matrix(NA, n.experiments, n.reps)
omega_record_rgdual <- matrix(NA, n.experiments, n.reps)
det_record_rgdual <- matrix(NA, n.experiments, n.reps)
iota_record_rgdual <- matrix(NA, n.experiments, n.reps)

lambda_record_apgffwd <- matrix(NA, n.experiments, n.reps)
gamma_record_apgffwd <- matrix(NA, n.experiments, n.reps)
omega_record_apgffwd <- matrix(NA, n.experiments, n.reps)
det_record_apgffwd <- matrix(NA, n.experiments, n.reps)
iota_record_apgffwd <- matrix(NA, n.experiments, n.reps)

nll_record_pco     <- matrix(NA, n.experiments, n.reps)
nll_record_rgdual  <- matrix(NA, n.experiments, n.reps)
nll_record_apgffwd <- matrix(NA, n.experiments, n.reps)

optim_iter_record_pco     <- matrix(NA, n.experiments, n.reps)
optim_iter_record_rgdual  <- matrix(NA, n.experiments, n.reps)
optim_iter_record_apgffwd <- matrix(NA, n.experiments, n.reps)

convergence_record_pco         <- matrix(NA, n.experiments, n.reps)
convergence_record_rgdual      <- matrix(NA, n.experiments, n.reps)
convergence_record_apgffwd     <- matrix(NA, n.experiments, n.reps)
convergence_msg_record_apgffwd <- vector(n.experiments * n.reps, mode='list')

for(i.experiment in 1:n.experiments) {
  # overwrite parameters
  # lambda <- test_vals[i.experiment]
  # omega <- test_vals[i.experiment]
  p <- test_vals[i.experiment]
  # rho.prior.strength <- test_vals[i.experiment]
  
  lambda_record_gen[i.experiment] <- lambda
  iota_record_gen  [i.experiment] <- iota
  gamma_record_gen [i.experiment] <- gamma
  omega_record_gen [i.experiment] <- omega
  p_record_gen     [i.experiment] <- p
  
  if(fix.lambda)
    fix.lambda.val <- lambda
  else
    fix.lambda.val <- FALSE
  if(fix.iota)
    fix.iota.val   <- iota
  else
    fix.iota.val   <- FALSE
  if(fix.gamma)
    fix.gamma.val  <- gamma
  else
    fix.gamma.val  <- FALSE
  if(fix.omega)
    fix.omega.val  <- omega
  else
    fix.omega.val  <- FALSE
  if(fix.rho)
    fix.rho.val    <- p
  else
    fix.rho.val    <- FALSE
  
  for(i.rep in 1:n.reps) {
    # sample data
    y <- matrix(NA, M, T)
    N <- matrix(NA, M, T)
    S <- matrix(NA, M, T-1)
    O <- matrix(NA, M, T-1)
    I <- matrix(NA, M, T-1)
    N[,1] <- rpois(M, lambda)
    for(t in 1:(T-1)) {
      # S[,t] <- rbinom(M, N[,t], omega)
      S[,t] <- 0
      O[,t] <- rpois(M, N[,t] * gamma)
      I[,t] <- rpois(M, iota)
      N[,t+1] <- S[,t] + O[,t] + I[,t]
    }
    y[] <- rbinom(M*T, N, p)
    Y = sum(y)
    y_record[[(i.experiment - 1) * n.reps + i.rep]] <- y
    
    # set K
    # ratio <- 0.386
    # ratio <- 0.2
    # ratio <- 1
    K <- K.default(y, p)
    K_record[i.experiment, i.rep] <- K
    
    # Prepare data
    umf <- unmarkedFramePCO(y = y, numPrimary=T)
    #summary(umf)
    
    if(do.pco == TRUE) {
      tryCatch({
        print(paste0('Starting trunc, val = ', test_vals[i.experiment], ', rep = ', i.rep))
        
        # # time model fit
        time.start <- proc.time()[3]
        
        # Fit model and backtransform
        # (m1 <- pcountOpen(~1, ~1, ~1, ~1, umf, K=K, immigration = immigration, dynamics = dynamics, se = FALSE)) # Typically, K should be higher
        (m1 <- pcountOpen_apgffwd(~1, ~1, ~1, ~1, umf, K = K, nll.fun = 'trunc', immigration = immigration, dynamics = dynamics, se = FALSE, method = pco_method, 
                                  # rho.prior.strength = rho.prior.strength,
                                  fix.rho = fix.rho.val, fix.iota = fix.iota.val, fix.gamma = fix.gamma.val, fix.omega = fix.omega.val, fix.lambda = fix.lambda.val))
    
        nll_record_pco[i.experiment, i.rep] <- m1@opt$value
        optim_iter_record_pco[i.experiment, i.rep] <- m1@opt$counts[1]
        convergence_record_pco[i.experiment, i.rep] <- m1@opt$convergence
        
        # (lam1 <- coef(backTransform(m1, "lambda"))) # or
        if(!is.null(coef(m1, type="lambda")))
          lambda_record_pco[i.experiment, i.rep] <- coef(backTransform(m1, "lambda"))
        if(!is.null(coef(m1, type="gamma")))
          gamma_record_pco[i.experiment, i.rep] <- coef(backTransform(m1, "gamma"))
        if(!is.null(coef(m1, type="omega")))
          omega_record_pco[i.experiment, i.rep] <- coef(backTransform(m1, "omega"))
        if(!is.null(coef(m1, type="det")))
          det_record_pco[i.experiment, i.rep] <- coef(backTransform(m1, "det"))
        if(!is.null(coef(m1, type="iota")))
          iota_record_pco[i.experiment, i.rep] <- coef(backTransform(m1, "iota"))
        # if(!is.null(coef(m1, type="lambda")))
        #   lam1 <- exp(coef(m1, type="lambda"))
        # if(!is.null(coef(m1, type="gamma")))
        #   gam1 <- exp(coef(m1, type="gamma"))
        # if(!is.null(coef(m1, type="omega")))
        #   om1 <- plogis(coef(m1, type="omega"))
        # if(!is.null(coef(m1, type="det")))
        #   p1 <- plogis(coef(m1, type="det"))
        # if(!is.null(coef(m1, type="iota")))
        #   i1 <- exp(coef(m1, type="iota"))
    
        runtime_unmarked[i.experiment, i.rep] <- proc.time()[3] - time.start
    
        print(paste0('rt = ', runtime_unmarked[i.experiment, i.rep]))
      }, error = function(e) {
        print(paste0('Something went wrong: ', e))
      })
    }
    
    if (do.rgdfwd == TRUE) {
      tryCatch({
        print(paste0('Starting rgdfwd, val = ', test_vals[i.experiment], ', rep = ', i.rep))
        
        time.start <- proc.time()[3]
    
        # Fit model and backtransform
        (m2 <- pcountOpen_rgdual(~1, ~1, ~1, ~1, umf, K=K, immigration = immigration, dynamics = dynamics)) # Typically, K should be higher
    
        nll_record_rgdual[i.experiment, i.rep] <- m2@opt$value
        optim_iter_record_rgdual[i.experiment, i.rep] <- m2@opt$counts[1]
        convergence_record_rgdual[i.experiment, i.rep] <- m2@opt$convergence
        
        if(!is.null(coef(m2, type="lambda")))
          lambda_record_rgdual[i.experiment, i.rep] <- coef(backTransform(m2, "lambda"))
        if(!is.null(coef(m2, type="gamma")))
          gamma_record_rgdual[i.experiment, i.rep] <- coef(backTransform(m2, "gamma"))
        if(!is.null(coef(m2, type="omega")))
          omega_record_rgdual[i.experiment, i.rep] <- coef(backTransform(m2, "omega"))
        if(!is.null(coef(m2, type="det")))
          det_record_rgdual[i.experiment, i.rep] <- coef(backTransform(m2, "det"))
        if(!is.null(coef(m2, type="iota")))
          iota_record_rgdual[i.experiment, i.rep] <- coef(backTransform(m2, "iota"))
        # (lam2 <- coef(backTransform(m2, "lambda"))) # or
        # if(!is.null(coef(m2, type="lambda")))
        #   lam2 <- exp(coef(m2, type="lambda"))
        # if(!is.null(coef(m2, type="gamma")))
        #   gam2 <- exp(coef(m2, type="gamma"))
        # if(!is.null(coef(m2, type="omega")))
        #   om2 <- plogis(coef(m2, type="omega"))
        # if(!is.null(coef(m2, type="det")))
        #   p2 <- plogis(coef(m2, type="det"))
    
        runtime_rgdfwd[i.experiment, i.rep] <- proc.time()[3] - time.start
        
        print(paste0('rt = ', runtime_rgdfwd[i.experiment, i.rep]))
      }, error = function(e) {
        print(paste0('Something went wrong: ', e))
      })
    }
    
    if (do.apgffwd == TRUE) {
      tryCatch({
        print(paste0('Starting apgffwd, val = ', test_vals[i.experiment], ', rep = ', i.rep))
        
        time.start <- proc.time()[3]
        
        # Fit model and backtransform
        (m3 <- pcountOpen_apgffwd(~1, ~1, ~1, ~1, umf, K=K, immigration = immigration, dynamics = dynamics, se= FALSE, method = apgffwd.method, 
                                  # rho.prior.strength = rho.prior.strength, 
                                  fix.rho = fix.rho.val, fix.iota = fix.iota.val, fix.gamma = fix.gamma.val, fix.omega = fix.omega.val, fix.lambda = fix.lambda.val)) # Typically, K should be higher
        
        nll_record_apgffwd[i.experiment, i.rep] <- m3@opt$value
        optim_iter_record_apgffwd[i.experiment, i.rep] <- m3@opt$counts[1]
        convergence_record_apgffwd[i.experiment, i.rep] <- m3@opt$convergence
        convergence_msg_record_apgffwd[[(i.experiment - 1) * n.reps + i.rep]] <- m3@opt$message
        
        if(!is.null(coef(m3, type="lambda"))) {
          if(identical(apgffwd.method, 'L-BFGS-B'))
            lambda_record_apgffwd[i.experiment, i.rep] <- coef(m3, type="lambda")
          else
            lambda_record_apgffwd[i.experiment, i.rep] <- coef(backTransform(m3, type="lambda"))
        }
        if(!is.null(coef(m3, type="gamma"))) {
          if(identical(apgffwd.method, 'L-BFGS-B'))
            gamma_record_apgffwd[i.experiment, i.rep] <- coef(m3, "gamma")
          else
            gamma_record_apgffwd[i.experiment, i.rep] <- coef(backTransform(m3, "gamma"))
        }
        if(!is.null(coef(m3, type="omega"))) {
          if(identical(apgffwd.method, 'L-BFGS-B'))
            omega_record_apgffwd[i.experiment, i.rep] <- coef(m3, "omega")
          else
            omega_record_apgffwd[i.experiment, i.rep] <- coef(backTransform(m3, "omega"))
        }
        # browser()
        if(!is.null(coef(m3, type="det"))) {
          if(identical(apgffwd.method, 'L-BFGS-B'))
            det_record_apgffwd[i.experiment, i.rep] <- coef(m3, "det")
          else
            det_record_apgffwd[i.experiment, i.rep] <- coef(backTransform(m3, "det"))
        }
        if(!is.null(coef(m3, type="iota"))) {
          if(identical(apgffwd.method, 'L-BFGS-B'))
            iota_record_apgffwd[i.experiment, i.rep] <- coef(m3, "iota")
          else
            iota_record_apgffwd[i.experiment, i.rep] <- coef(backTransform(m3, "iota"))
        }
        
        # if(!is.null(coef(m3, type="iota")))
        #   lamda_record_apgffwd[i.experiment, i.rep] <- coef(backTransform(m3, "iota"))
        # if(!is.null(coef(m3, type="gamma")))
        #   gamma_record_apgffwd[i.experiment, i.rep] <- coef(backTransform(m3, "gamma"))
        # if(!is.null(coef(m3, type="omega")))
        #   omega_record_apgffwd[i.experiment, i.rep] <- coef(backTransform(m3, "omega"))
        # if(!is.null(coef(m3, type="det")))
        #   det_record_apgffwd[i.experiment, i.rep] <- coef(backTransform(m3, "det"))
        # if(!is.null(coef(m3, type="iota")))
        #   iota_record_apgffwd[i.experiment, i.rep] <- coef(backTransform(m3, "iota"))
        # (lam3 <- coef(backTransform(m3, "lambda"))) # or
        # if(!is.null(coef(m3, type="lambda")))
        #   lam3 <- exp(coef(m3, type="lambda"))
        # if(!is.null(coef(m3, type="gamma")))
        #   gam3 <- exp(coef(m3, type="gamma"))
        # if(!is.null(coef(m3, type="omega")))
        #   om3 <- plogis(coef(m3, type="omega"))
        # if(!is.null(coef(m3, type="det")))
        #   p3 <- plogis(coef(m3, type="det"))
        # if(!is.null(coef(m3, type="iota")))
        #   i3 <- exp(coef(m3, type="iota"))
        
        runtime_apgffwd[i.experiment, i.rep] <- proc.time()[3] - time.start
        print(paste0('rt = ', runtime_apgffwd[i.experiment, i.rep]))
      }, error = function(e) {
        print(paste0('Something went wrong: ', e))
      })
    }
    
    # browser()
  }
}
}