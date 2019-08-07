source("pcountOpen_constrained_apgffwd.R")
# source("pcountOpen")

library(ggplot2)

inv.logit <- function(x) { exp(x) / (exp(x) + 1) }
logit <- function(x) { log(x) - log(1 - x) }

M <- 2
T <- 5
lambda <- 15
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
  # S[,t] <- 0
  O[,t] <- rpois(M, N[,t] * gamma)
  I[,t] <- rpois(M, iota)
  N[,t+1] <- S[,t] + O[,t] + I[,t]
}
y[] <- rbinom(M*T, N, p)
Y = sum(y)

dynamics    = "autoreg"
immigration = TRUE

arrival.pgf <- pgf.poisson
offspring.pgf   <- function(s, theta) {
  return(pgf.bernoulli(s, theta['p']) * pgf.poisson(s, theta['lambda']))
}

umf <- unmarkedFramePCO(y = y, numPrimary=T)

eval.param <- c("iota")
eval.vals <- seq(0.05, 0.95, 0.05)
if(length(eval.param) > 1) {
  title <- paste0("nll vs arrival params")
} else {
  title <- paste0("nll vs ", eval.param)
}

# eval.params <- c(1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 30, 50, 100)
# eval.params <- 0.5
nll.apgf <- rep(0, length(eval.vals))
nll.fwd  <- rep(0, length(eval.vals))
nll.apgfpco <- rep(0, length(eval.vals))
nll.truncpco <- rep(0, length(eval.vals))
for(i.ev in 1:length(eval.vals)) {
  ev <- eval.vals[i.ev]
  
  if ("lambda" %in% eval.param) {
    lambda.eval <- ev
  } else {
    lambda.eval <- lambda
  }
  
  if ("gamma" %in% eval.param) {
    gamma.eval <- ev
  } else {
    gamma.eval <- gamma
  }
  
  if ("omega" %in% eval.param) {
    omega.eval <- ev
  } else {
    omega.eval <- omega
  }
  
  if ("det" %in% eval.param || "p" %in% eval.param) {
    det.eval <- ev
  } else {
    det.eval <- det
  }
  
  if ("iota" %in% eval.param) {
    iota.eval <- ev
  } else {
    iota.eval <- iota
  }
  
  eval.at <- c(lambda.eval, gamma.eval, omega.eval, det.eval, iota.eval)
  nll.apgfpco[i.ev] <- pcountOpen_apgffwd(~1, ~1, ~1, ~1, umf, immigration = immigration, dynamics = dynamics, eval.at = eval.at, nll.fun = "rgd")
  eval.at.trunc <- c(log(lambda.eval), log(gamma.eval), logit(omega.eval), logit(det.eval), log(iota.eval))
  nll.truncpco[i.ev] <- pcountOpen_apgffwd(~1, ~1, ~1, ~1, umf, immigration = immigration, dynamics = dynamics, eval.at = eval.at.trunc, nll.fun = "trunc")
  
  theta.arrival <- data.frame(lambda = c(lambda.eval, rep(iota.eval, T - 1)))
  # theta.offspring <- data.frame(lambda = rep(gamma.eval, T - 1))
  theta.offspring <- data.frame(p      = array(omega.eval, T - 1),
                                lambda = array(gamma.eval, T - 1))
  rho <- data.frame(p = rep(det.eval, T))
  # browser()
  for(i.y in 1:nrow(y)) {
    nll.apgf.iter <- apgffwd(y[i.y,], arrival.pgf, theta.arrival, offspring.pgf, theta.offspring, rho)
    # browser()
    if(nll.apgf.iter$sign < 0)
      browser()
    nll.apgf[i.ev] <- nll.apgf[i.ev] - nll.apgf.iter$mag
    nll.fwd[i.ev] <- nll.fwd[i.ev] - forward.ll(forward(y[i.y,], arrival.pgf, theta.arrival, offspring.pgf, theta.offspring, rho))
  }
}

plot.df <- data.frame(x = eval.vals, nll.apgf = nll.apgfpco, nll.fwd = nll.fwd, nll.truncpco)
g <- ggplot(plot.df, aes()) + 
  geom_point(aes(x = eval.vals, y = nll.apgf, col = "apgffwd"), shape = 0) + 
  geom_point(aes(x = eval.vals, y = nll.fwd,  col = "gdfwd"),   shape = 1) +
  geom_point(aes(x = eval.vals, y = nll.truncpco,  col = "trunc"),   shape = 2)
plot(g)

# plot(eval.params, do.call(rbind, lapply(nll.apgf, as.numeric)), ylab = "nll", xlab = "eval.param")