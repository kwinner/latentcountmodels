library(unmarked)
source('pcountOpen_rgdual.R')

# test_vals <- seq(5, 95, 10)
# test_vals <- seq(0.05, 0.95, 0.5)
test_vals <- 0.5
n.experiments <- length(test_vals)
n.reps <- 1

runtime_unmarked <- matrix(NA, n.experiments, n.reps)
runtime_rgdfwd   <- matrix(NA, n.experiments, n.reps)
Y_record <- matrix(NA, n.experiments, n.reps)
K_record <- matrix(NA, n.experiments, n.reps)

for(i.experiment in 1:n.experiments) {
  for(i.rep in 1:n.reps) {
    M <- 2
    T <- 5
    lambda <- 100
    #lambda <- test_vals[i.experiment]
    gamma <- 1.5
    delta <- 0.8
    omega <- test_vals[i.experiment]
    p <- 0.7
    y <- N <- matrix(NA, M, T)
    S <- G <- matrix(NA, M, T-1)
    N[,1] <- rpois(M, lambda)
    for(t in 1:(T-1)) {
      S[,t] <- rbinom(M, N[,t], delta)
      G[,t] <- rpois(M, gamma)
      N[,t+1] <- S[,t] + G[,t]
    }
    y[] <- rbinom(M*T, N, p)
    Y = sum(y)
    Y_record[i.experiment, i.rep] <- Y
    
    dynamics    = "trend"
    immigration = TRUE
    
    # set K
    ratio <- 0.38563049853372433
    # ratio <- 0.2
    # ratio <- 1
    K <- ceiling((ratio * Y + 10) / omega)
    K_record[i.experiment, i.rep] <- K
    
    # Prepare data
    umf <- unmarkedFramePCO(y = y, numPrimary=T)
    #summary(umf)
    
    # # time model fit
    time.start <- proc.time()[3]

    # Fit model and backtransform
    (m1 <- pcountOpen(~1, ~1, ~1, ~1, umf, K=K, immigration = immigration, dynamics = dynamics)) # Typically, K should be higher

    (lam <- coef(backTransform(m1, "lambda"))) # or
    if(!is.null(coef(m1, type="lambda")))
      lam <- exp(coef(m1, type="lambda"))
    if(!is.null(coef(m1, type="gamma")))
      gam <- exp(coef(m1, type="gamma"))
    if(!is.null(coef(m1, type="omega")))
      om <- plogis(coef(m1, type="omega"))
    if(!is.null(coef(m1, type="det")))
      p <- plogis(coef(m1, type="det"))

    runtime_unmarked[i.experiment, i.rep] <- proc.time()[3] - time.start
    
    time.start <- proc.time()[3]
    
    # Fit model and backtransform
    (m1 <- pcountOpen_rgdual(~1, ~1, ~1, ~1, umf, K=K, immigration = immigration, dynamics = dynamics)) # Typically, K should be higher
    
    (lam <- coef(backTransform(m1, "lambda"))) # or
    if(!is.null(coef(m1, type="lambda")))
      lam <- exp(coef(m1, type="lambda"))
    if(!is.null(coef(m1, type="gamma")))
      gam <- exp(coef(m1, type="gamma"))
    if(!is.null(coef(m1, type="omega")))
      om <- plogis(coef(m1, type="omega"))
    if(!is.null(coef(m1, type="det")))
      p <- plogis(coef(m1, type="det"))
    
    runtime_rgdfwd[i.experiment, i.rep] <- proc.time()[3] - time.start
  }
}