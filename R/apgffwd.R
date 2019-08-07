source("rgdual.R")

##################################################################
#                      APGFFWD ALGORITHM                        #
##################################################################

# get parameters of a Poisson dist'n with specified mean (this one is trivial...)
inv.poiss <- function(mean.p) {
  return(c(lambda = mean.p))
}

# get parameters of a NB dist'n with specified mean and variance
inv.nb <- function(mean.p, var.p) {
  stopifnot(var.p > mean.p)

  r.q <- (mean.p ^ 2) / (var.p - mean.p)
  p.q <- mean.p / var.p

  return(c(r = r.q, p = p.q))
}

# get parameters of a Binomial dist'n with specified mean and variance
inv.binom <- function(mean.p, var.p, y = NA) {
  stopifnot(mean.p > var.p)

  n.q <- mean.p / (1 - (var.p / mean.p))

  # note: for a binomial, n.q must be integral
  n.q <- round(n.q)
  
  # optionally lower bound n.q by y
  if (!is.na(y))
    n.q <- max(n.q, y)
  
  p.q <- mean.p / n.q

  return(c(n = n.q, p = p.q))
}

# returns a parametric PGF that approximates F.pgf by matching the first two moments
apgf <- function(F.pgf, logZ = NA, 
                 return.distn = FALSE, return.param = FALSE, return.pgf = TRUE,
                 debug = FALSE, y = NA) {
  # compute the normalization constant for F.pgf if it wasn't provided
  if (any(is.na(logZ))) {
    logZ <- F.pgf(lsgd.1dx(q=1))
  }

  # renormalize the dist'n before computing moments (this is now done later on)
  # Fstar.pgf <- function(s) { F.pgf(s) / exp(logZ) }

  # construct F.mgf, the MGF of p from F.pgf
  F.mgf <- function(t) { F.pgf(exp(t))}
  
  if (debug == TRUE)
    browser()
  
  # use the MGF to compute the first two moments of p
  moments.p <- exp(F.mgf(lsgd.xdx(0, q = 3))$mag[2:3] + lgamma(2:3) - logZ$mag)
  
  mean.p <- moments.p[1]
  var.p  <- moments.p[2] - (mean.p ^ 2)
  
  if (is.nan(mean.p) || is.nan(var.p))
    browser()
  
  # if (var.p == 0)
  #   browser()
  # print(mean.p)
  # print(var.p)
  
  if (debug == TRUE)
    browser()
  
  return.list <- list()
  # if p.mean and p.var are approximately equal (helps with stability)
  if (abs(mean.p - var.p) <= 1e-6) {
    distn    <- 'poiss'
    theta.q  <- inv.poiss(mean.p)
    if (return.pgf)
      return.list$pgf <- function(s, theta = theta.q, logZ = logZ) { exp(logZ) * pgf.poisson(s, theta) }
  } else if(mean.p < var.p) {
    if (debug == TRUE)
      browser()
    distn    <- 'nb'
    theta.q  <- inv.nb(mean.p, var.p)
    if (return.pgf)
      return.list$pgf <- function(s, theta = theta.q, logZ = logZ) { exp(logZ) * pgf.negbin(s, theta) }
  } else if(mean.p > var.p) {
    if (debug == TRUE)
      browser()
    distn    <- 'binom'
    theta.q  <- inv.binom(mean.p, var.p)
    # if (theta.q['n'] < y)
    #   stop("Unable to approximate PGF with mean ", mean.p, " and var ", var.p)
    if (return.pgf)
      return.list$pgf <- function(s, theta = theta.q, logZ = logZ) { exp(logZ) * pgf.binom(s, theta) }
  }
  
  if (return.distn)
    return.list$distn <- distn
  if (return.param)
    return.list$param <- theta.q

  if (length(return.list) == 1)
    return(return.list[[1]])
  else
    return(return.list)
}

# returns a parametric PGF that approximates F.pgf by matching the first two moments
apgf2 <- function(F.pgf, logZ = NA, 
                 return.distn = FALSE, return.param = FALSE, return.pgf = TRUE,
                 debug = FALSE, y = NA) {
  # compute the normalization constant for F.pgf if it wasn't provided
  if (any(is.na(logZ))) {
    logZ <- F.pgf(lsgd.1dx(q=1))
  }
  
  # renormalize the dist'n before computing moments (this is now done later on)
  # Fstar.pgf <- function(s) { F.pgf(s) / exp(logZ) }
  
  # construct F.mgf, the MGF of p from F.pgf
  F.mgf <- function(t) { F.pgf(exp(t))}
  
  if (debug == TRUE)
    browser()
  
  # use the MGF to compute the first two moments of p
  moments.p <- exp(F.mgf(lsgd.xdx(0, q = 3))$mag[2:3] + lgamma(2:3) - logZ$mag)
  moments2.p <- as.ls(F.mgf(lsgd.xdx(0, q = 3))[2:3,]) * as.ls.magsign(mag = lgamma(2:3)) / as.ls.magsign(mag = logZ)
  
  mean.p <- moments.p[1]
  var.p  <- moments.p[2] - (mean.p ^ 2)
  
  mean2.p <- moments2.p[1,]
  var2.p <- moments2.p[2,] - (mean2.p ^ 2)
  
  if (debug == TRUE)
    browser()
  
  if (is.nan(mean.p) || is.nan(var.p)) {
    # browser()
    stop("Unable to approximate PGF, mean or variance are NaN")
  }
  
  
  # if (var.p == 0)
  #   browser()
  # print(mean.p)
  # print(var.p)
  
  if (debug == TRUE)
    browser()
  
  return.list <- list()
  # if p.mean and p.var are approximately equal (helps with stability)
  if (abs(mean.p - var.p) <= 1e-6) {
    distn    <- 'poiss'
    theta.q  <- inv.poiss(mean.p)
    if (return.pgf)
      return.list$pgf <- function(s, theta = theta.q, logZ = logZ) { exp(logZ) * pgf.poisson(s, theta) }
  } else if(mean.p < var.p) {
    if (debug == TRUE)
      browser()
    distn    <- 'nb'
    theta.q  <- inv.nb(mean.p, var.p)
    if (return.pgf)
      return.list$pgf <- function(s, theta = theta.q, logZ = logZ) { exp(logZ) * pgf.negbin(s, theta) }
  } else if(mean.p > var.p) {
    if (debug == TRUE)
      browser()
    distn    <- 'binom'
    theta.q  <- inv.binom(mean.p, var.p)
    if (theta.q['n'] < y)
      stop("Unable to approximate PGF with mean ", mean.p, " and var ", var.p)
    if (return.pgf)
      return.list$pgf <- function(s, theta = theta.q, logZ = logZ) { exp(logZ) * pgf.binom(s, theta) }
  }
  
  if (return.distn)
    return.list$distn <- distn
  if (return.param)
    return.list$param <- theta.q
  
  if (length(return.list) == 1)
    return(return.list[[1]])
  else
    return(return.list)
}

apgffwd <- function(y,
                    arrival.pgf,
                    theta.arrival,
                    branch.pgf,
                    theta.branch,
                    theta.observ,
                    d = 0,
                    debug = FALSE) {
  # print(y)
  # print(theta.arrival)
  # print(theta.branch)
  # print(theta.observ)

  # if (theta.arrival[1,] >= 1.0e54) {
  #   debug = TRUE
  #   browser()
  # }
  
  K <- length(y)

  Gamma.list           <- vector(mode="list", length=K)
  hat.Gamma.distn.list <- vector(mode="list", length=K)
  hat.Gamma.param.list <- vector(mode="list", length=K)
  Gamma.logZ.list      <- vector(mode="list", length=K)
  Alpha.list           <- vector(mode="list", length=K)

  # # wrap the arrival/branch pgfs, which may each be a list of pgfs
  # # note: if arrival.pgf/branch.pgf are lists, they need to be length K or K - 1 accordingly
  # if (is.list(arrival.pgf))
  #   arrival.pgf.select <- function(i) { function(t, k = i) { arrival.pgf[k](t, theta.arrival[k,,drop=F]) } }
  # else
  #   arrival.pgf.select <- function(i) { function(t, k = i) { arrival.pgf(t, theta.arrival[k,,drop=F]) } }
  # if (is.list(branch.pgf))
  #   branch.pgf.select <- function(i) { function(t, k = i) { branch.pgf[k](t, theta.branch[k,,drop=F]) } }
  # else
  #   branch.pgf.select <- function(i) { function(t, k = i) { branch.pgf(t, theta.branch[k,,drop=F]) } }

  # main loop
  for (i in 1:K) {
    # define the Gamma message
    if (i == 1)
      # Gamma.list[[i]] <- function(u, k = i) { arrival.pgf.select(k)(u) }
      Gamma.list[[i]] <- function(u, k = i) { arrival.pgf(u, theta.arrival[k,,drop=FALSE]) }
    else
      # Gamma.list[[i]] <- function(u, k = i) { Alpha.list[[k-1]](branch.pgf.select(k-1)(u), k-1) * arrival.pgf.select(k)(u) }
      Gamma.list[[i]] <- function(u, k = i) { Alpha.list[[k-1]](branch.pgf(u, theta.branch[k-1,,drop=FALSE]), k-1) * arrival.pgf(u, theta.arrival[k,,drop=FALSE]) }

    if (debug == TRUE)
      browser()
    
    # approximate the Gamma message
    Gamma.logZ.list[[i]] <- Gamma.list[[i]](lsgd.1dx(q=1))[1,]
    hat.Gamma <- apgf2(Gamma.list[[i]], Gamma.logZ.list[[i]], return.distn = TRUE, return.param = TRUE, return.pgf = FALSE, debug = debug, y = y[i])
    hat.Gamma.distn.list[[i]] <- hat.Gamma$distn
    hat.Gamma.param.list[[i]] <- hat.Gamma$param
    
    if (hat.Gamma.distn.list[[i]] == 'poiss')
      Alpha.list[[i]] <- function(s, k) {
        lambda <- hat.Gamma.param.list[[k]]['lambda']
        as.ls.magsign(Gamma.logZ.list[[k]]$mag + y[k] * (log(lambda) + log(theta.observ[k,])) - lambda - lgamma(y[k] + 1)) *
          (s ^ y[k]) * exp(lambda * (1 - theta.observ[k,]) * s)
      }
    else if (hat.Gamma.distn.list[[i]] == 'nb')
      Alpha.list[[i]] <- function(s, k) {
        r <- hat.Gamma.param.list[[k]]['r']
        p <- hat.Gamma.param.list[[k]]['p']
        as.ls.magsign(Gamma.logZ.list[[k]]$mag + y[k] * (log(1 - p) + log(theta.observ[k,])) + r * log(p) + lgamma(r + y[k]) - lgamma(y[k] + 1) - lgamma(r)) *
          (s ^ y[k]) * ((1 - (1 - theta.observ[k,]) * (1 - p) * s) ^ (-r - y[k]))
      }
    else if (hat.Gamma.distn.list[[i]] == 'binom')
      Alpha.list[[i]] <- function(s, k) {
        n <- hat.Gamma.param.list[[k]]['n']
        p <- hat.Gamma.param.list[[k]]['p']
        as.ls.magsign(Gamma.logZ.list[[k]]$mag + y[k] * (log(p) + log(theta.observ[k,])) + lgamma(n + 1) - lgamma(y[k] + 1) - lgamma(n - y[k] + 1)) *
          (s ^ y[k]) * ((1 - p + (p * (1 - theta.observ[k,]) * s)) ^ (n - y[k]))
      }
    if (debug == TRUE)
      browser()
  }

  val <- Alpha.list[[length(Alpha.list)]](1.0, k = length(Alpha.list))
  
  # print(val)
  
  return(val)
}

# y      <- 20*c(1,2,3,1,3)
# lambda <- 20*data.frame(lambda = c(2.5, 6, 6, 6, 6))
# delta  <- data.frame(lambda = c(0.5, 0.5, 0.5, 0.5))
# rho    <- data.frame(p      = c(0.2, 0.2, 0.2, 0.2, 0.2))
# rho    <- data.frame(p      = c(1.0, 1.0, 1.0, 1.0, 1.0))

# delta  <- data.frame(p = c(0.5, 0.5, 0.5, 0.5))

# y      <- c(79,72,46,37,35)
# lambda <- data.frame(lambda = c(5.75e68, 7.14e-1, 7.14e-1, 7.14e-1, 7.14e-1))
# delta  <- data.frame(lambda = c(2.01e-16, 2.01e-16, 2.01e-16, 2.01e-16))
# rho    <- data.frame(p      = c(1.0, 1.0, 1.0, 1.0, 1.0))

# pgf.poisson    <- function(s, theta) {
#   browser()
#   return(exp(theta$lambda * (s - 1)))
# }

# F <- function(s) { pgf.poisson(s, lambda[1,, drop = FALSE]) }
# print(apgf(F, return.distn = TRUE, return.param = TRUE, return.pgf = FALSE, debug = TRUE))

# s.lsgd <- lsgd.xdx(1, q = 3)
# fact.moments <- F(s.lsgd)[2:3,]$mag + lgamma(2:3)
# mean.F <- exp(fact.moments[1])
# var.F <- exp(fact.moments[2]) + mean.F - (mean.F ^ 2)
# print(mean.F)
# print(var.F)

# A <- forward(y, pgf.poisson, lambda, pgf.poisson, delta, rho)
# B <- apgffwd(y, pgf.poisson, lambda, pgf.poisson, delta, rho)
# 
# print(forward.ll(A))
# print(B)