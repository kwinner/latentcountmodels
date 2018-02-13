library(unmarked)
source("pcountOpen_ngdual.R")

logit <- function(p) {
  return(log(p) - log(1 - p))
}

logistic <- function(x) {
  return(1.0/(1.0 + exp(-x)))
}

# test_vals <- seq(5, 95, 10)
test_vals <- 50
# test_vals <- seq(0.05, 0.95, 0.1)
n.experiments <- length(test_vals)
n.reps <- 10

runtime_unmarked <- matrix(NA, n.experiments, n.reps)
runtime_ngdualf  <- matrix(NA, n.experiments, n.reps)
Y_record <- matrix(NA, n.experiments, n.reps)
K_record <- matrix(NA, n.experiments, n.reps)

for(i.experiment in 1:n.experiments) {
  for(i.rep in 1:n.reps) {
    M <- 1
    T <- 5
    # lambda <- 50
    lambda <- test_vals[i.experiment]
    gamma <- 10
    omega <- 0.5
    # omega <- test_vals[i.experiment]
    p <- 0.7
    
    y <- N <- matrix(NA, M, T)
    S <- G <- matrix(NA, M, T-1)
    N[,1] <- rpois(M, lambda)
    for(t in 1:(T-1)) {
      S[,t] <- rbinom(M, N[,t], omega)
      G[,t] <- rpois(M, gamma)
      N[,t+1] <- S[,t] + G[,t]
    }
    y[] <- rbinom(M*T, N, p)
    Y = sum(y)
    Y_record[i.experiment, i.rep] <- Y
    
    # set K
    ratio <- 0.38563049853372433
    K <- ceiling((ratio * Y + 10) / omega)
    K_record[i.experiment, i.rep] <- K
    
    # Prepare data
    umf <- unmarkedFramePCO(y = y, numPrimary=T)
    #summary(umf)
    
    # # time model fit
    # time.start <- proc.time()[3]
    
    # # Fit model and backtransform
    # (m1 <- pcountOpen(~1, ~1, ~1, ~1, umf, K=K, immigration = TRUE, dynamics = "trend")) # Typically, K should be higher
    
    # # (lam <- coef(backTransform(m1, "lambda"))) # or
    # # lam <- exp(coef(m1, type="lambda"))
    # # gam <- exp(coef(m1, type="gamma"))
    # # om <- plogis(coef(m1, type="omega"))
    # # p <- plogis(coef(m1, type="det"))
    # if("lambda" %in% m1@estimates@estimates) {
    #     lam <- exp(m1@estimates@estimates$lambda)
    # }
    # if("gamma" %in% m1@estimates@estimates) {
    #     gam <- exp(m1@estimates@estimates$gamma)
    # }
    # if("omega" %in% m1@estimates@estimates) {
    #     om <- plogis(m1@estimates@estimates$omega)
    # }
    # if("det" %in% m1@estimates@estimates) {
    #     p <- plogis(m1@estimates@estimates$det)
    # }
    # if("iota" %in% m1@estimates@estimates) {
    #     iota <- m1@estimates@estimates$iota
    # }
    
    # runtime_unmarked[i.experiment, i.rep] <- proc.time()[3] - time.start
    
    time.start <- proc.time()[3]

    # Fit model and backtransform
    (m1 <- pcountOpen_ngdual(~1, ~1, ~1, ~1, umf, K=K, immigration = TRUE, dynamics = "trend", starts = c(log(lambda), logit(omega), logit(p), log(gamma)))) # Typically, K should be higher
    
    # (lam <- coef(backTransform(m1, "lambda"))) # or
    # lam <- exp(coef(m1, type="lambda"))
    # gam <- exp(coef(m1, type="gamma"))
    # om <- plogis(coef(m1, type="omega"))
    # p <- plogis(coef(m1, type="det"))

    if("lambda" %in% m1@estimates@estimates) {
        lam <- exp(m1@estimates@estimates$lambda)
    }
    if("gamma" %in% m1@estimates@estimates) {
        gam <- exp(m1@estimates@estimates$gamma)
    }
    if("omega" %in% m1@estimates@estimates) {
        om <- plogis(m1@estimates@estimates$omega)
    }
    if("det" %in% m1@estimates@estimates) {
        p <- plogis(m1@estimates@estimates$det)
    }
    if("iota" %in% m1@estimates@estimates) {
        iota <- m1@estimates@estimates$iota
    }
        
    runtime_ngdualf[i.experiment, i.rep] <- proc.time()[3] - time.start
  }
}