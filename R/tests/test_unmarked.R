library(unmarked)
source('pcountOpen_rgdual.R')
source('pcountOpen_apgffwd.R')

do.pco     = TRUE
do.rgdfwd  = TRUE
do.apgffwd = TRUE

test_vals <- seq(15, 95, 10)
# test_vals <- seq(0.05, 0.95, 0.1)
# test_vals <- 0.5
n.experiments <- length(test_vals)
n.reps <- 5

runtime_unmarked <- matrix(NA, n.experiments, n.reps)
runtime_rgdfwd   <- matrix(NA, n.experiments, n.reps)
runtime_apgffwd  <- matrix(NA, n.experiments, n.reps)
Y_record <- matrix(NA, n.experiments, n.reps)
K_record <- matrix(NA, n.experiments, n.reps)

lamda_record_pco <- matrix(NA, n.experiments, n.reps)
gamma_record_pco <- matrix(NA, n.experiments, n.reps)
omega_record_pco <- matrix(NA, n.experiments, n.reps)
det_record_pco <- matrix(NA, n.experiments, n.reps)
iota_record_pco <- matrix(NA, n.experiments, n.reps)

lamda_record_rgdual <- matrix(NA, n.experiments, n.reps)
gamma_record_rgdual <- matrix(NA, n.experiments, n.reps)
omega_record_rgdual <- matrix(NA, n.experiments, n.reps)
det_record_rgdual <- matrix(NA, n.experiments, n.reps)
iota_record_rgdual <- matrix(NA, n.experiments, n.reps)

lamda_record_apgffwd <- matrix(NA, n.experiments, n.reps)
gamma_record_apgffwd <- matrix(NA, n.experiments, n.reps)
omega_record_apgffwd <- matrix(NA, n.experiments, n.reps)
det_record_apgffwd <- matrix(NA, n.experiments, n.reps)
iota_record_apgffwd <- matrix(NA, n.experiments, n.reps)

for(i.experiment in 1:n.experiments) {
  for(i.rep in 1:n.reps) {
    M <- 2
    T <- 5
    # lambda <- 10
    lambda <- test_vals[i.experiment]
    iota <- 10
    gamma <- 0.2
    # omega <- 0.8
    # omega <- test_vals[i.experiment]
    omega <- 0.4
    p <- 0.7
    y <- matrix(NA, M, T)
    N <- matrix(NA, M, T)
    S <- matrix(NA, M, T-1)
    O <- matrix(NA, M, T-1)
    I <- matrix(NA, M, T-1)
    N[,1] <- rpois(M, lambda)
    for(t in 1:(T-1)) {
      S[,t] <- rbinom(M, N[,t], omega)
      O[,t] <- rpois(M, N[,t] * gamma)
      I[,t] <- rpois(M, iota)
      N[,t+1] <- S[,t] + O[,t] + I[,t]
    }
    y[] <- rbinom(M*T, N, p)
    Y = sum(y)
    Y_record[i.experiment, i.rep] <- Y
    
    dynamics    = "autoreg"
    immigration = TRUE
    
    # set K
    ratio <- 0.38563049853372433
    # ratio <- 0.2
    # ratio <- 1
    K <- ceiling((ratio * Y + 10) / p)
    K_record[i.experiment, i.rep] <- K
    
    # Prepare data
    umf <- unmarkedFramePCO(y = y, numPrimary=T)
    #summary(umf)
    
    if(do.pco == TRUE) {
      print(paste0('Starting trunc, val = ', test_vals[i.experiment], ', rep = ', i.rep))
  
      # # time model fit
      time.start <- proc.time()[3]
  
      # Fit model and backtransform
      (m1 <- pcountOpen(~1, ~1, ~1, ~1, umf, K=K, immigration = immigration, dynamics = dynamics, se = FALSE)) # Typically, K should be higher
  
      # (lam1 <- coef(backTransform(m1, "lambda"))) # or
      if(!is.null(coef(m1, type="lambda")))
        lamda_record_pco[i.experiment, i.rep] <- coef(backTransform(m1, "lambda"))
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
    }
    
    if (do.rgdfwd == TRUE) {
      print(paste0('Starting rgdfwd, val = ', test_vals[i.experiment], ', rep = ', i.rep))
      
      time.start <- proc.time()[3]
  
      # Fit model and backtransform
      (m2 <- pcountOpen_rgdual(~1, ~1, ~1, ~1, umf, K=K, immigration = immigration, dynamics = dynamics)) # Typically, K should be higher
  
      if(!is.null(coef(m2, type="lambda")))
        lamda_record_rgdual[i.experiment, i.rep] <- coef(backTransform(m2, "lambda"))
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
    }
    
    if (do.apgffwd == TRUE) {
      print(paste0('Starting apgffwd, val = ', test_vals[i.experiment], ', rep = ', i.rep))
      
      time.start <- proc.time()[3]
      
      # Fit model and backtransform
      (m3 <- pcountOpen_apgffwd(~1, ~1, ~1, ~1, umf, K=K, immigration = immigration, dynamics = dynamics, se= FALSE, method = 'L-BFGS-B')) # Typically, K should be higher
      
      if(!is.null(coef(m3, type="lambda")))
        lamda_record_apgffwd[i.experiment, i.rep] <- coef(backTransform(m3, "lambda"))
      if(!is.null(coef(m3, type="gamma")))
        gamma_record_apgffwd[i.experiment, i.rep] <- coef(backTransform(m3, "gamma"))
      if(!is.null(coef(m3, type="omega")))
        omega_record_apgffwd[i.experiment, i.rep] <- coef(backTransform(m3, "omega"))
      if(!is.null(coef(m3, type="det")))
        det_record_apgffwd[i.experiment, i.rep] <- coef(backTransform(m3, "det"))
      if(!is.null(coef(m3, type="iota")))
        iota_record_apgffwd[i.experiment, i.rep] <- coef(backTransform(m3, "iota"))
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
    }
    
    # browser()
  }
}