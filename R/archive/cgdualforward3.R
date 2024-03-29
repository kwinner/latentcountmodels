source("cgdual.R")

gdualforward <- function(y,
                         arrival.pgf.lsgd,
                         theta.arrival,
                         branch.pgf.lsgd,
                         theta.branch,
                         theta.observ,
                         d = 1) {
  K <- length(y)
  
  Alpha <- vector(mode = "list", length = K)
  
  liftGamma <- function(u, k) {
    if(k <= 1)
      return(gdual.mul(lsgdual.1dx(length(u$mag)),
                       arrival.pgf.lsgd(u, theta.arrival[k])))
    else
      return(gdual.mul(liftA(branch.pgf.lsgd(u, theta.branch[k-1]), k - 1),
                       arrival.pgf.lsgd(u, theta.arrival[k])))
    
  }
  
  # recursive function for computing the Alpha messages
  # s_ds : value to evaluate A_k() at
  # k    : current index
  # q_k  : requested length of A_k message
  #
  # return <alpha_k, ds_k>_{q_k}
  liftA <- function(s,
                    k) {
    q_k <- length(s$mag)
    
    #recursion base case now handled in liftGamma    
    # if(k < 1)
    #   return(lsgdual.1dx(q_k))
    
    # note: possible/likely optimization from operations w/ scalars/constants
    # const <- gdual.div(gdual.pow(gdual.mul(s, lsgdual.cdx(theta.observ[k], q_k)),
    #                              y[k]),
    #                    lsgdual.cdx.ls(ls(mag=lgamma(y[k] + 1), sign = 1), q_k))
    
    const1 <- gdual.pow(s, y[k])
    const2 <- lsgdual.cdx.ls(ls(mag = (y[k] * log(theta.observ[k])) - lgamma(y[k] + 1), sign = 1), q_k)
    const  <- gdual.mul(const1, const2)
    
    Gamma_k <- function(u_k) { liftGamma(u_k, k) }
    alpha <- gdual.diff(Gamma_k,
                        gdual.mul(s, lsgdual.cdx(1 - theta.observ[k], q_k)),
                        y[k])
    alpha   <- gdual.mul(alpha, const)
    
    Alpha[[k]] <<- alpha
    
    return(alpha)
  }
  
  gdual.diff((function(s) liftA(s, K)), 1.0, d)
  
  return(Alpha)
}

poisson.lsgd <- function(s, theta) {
  q <- length(s$mag)
  lambda <- theta[1]
  
  return(gdual.exp(gdual.mul(gdual.add(s, lsgdual.cdx(-1.0, q)),
                             lsgdual.cdx(lambda, q))))
}

bernoulli.lsgd <- function(s, theta) {
  q <- length(s$mag)
  p <- theta[1]
  
  return(gdual.add(gdual.mul(s, lsgdual.cdx(p, q)),
                   lsgdual.cdx(1 - p, q)))
}

binomial.lsgd <- function(s, theta) {
  q <- length(s$mag)
  n <- theta[1]
  p <- theta[2]
  
  return(gdual.pow(gdual.add(gdual.mul(s, lsgdual.cdx(p, q)),
                             lsgdual.cdx(1 - p, q)),
                   n))
}

negbin.lsgd <- function(s, theta) {
  q <- length(s$mag)
  r <- theta[1]
  p <- theta[2]
  
  return(gdual.pow(gdual.mul(gdual.inv(gdual.add(gdual.mul(s, lsgdual.cdx(p - 1, q)),
                                                 lsgdual.1dx(q))),
                             lsgdual.cdx(p, q)),
                   r))
}

logarithmic.lsgd <- function(s, theta) {
  q <- length(s$mag)
  p <- theta[1]
  
  return(gdual.mul(gdual.log(gdual.add(gdual.mul(s, lsgdual.cdx(-p, q)),
                                       lsgdual.1dx(q))),
                   lsgdual.cdx(1.0 / log(1 - p), q)))
}

geometric.lsgd <- function(s, theta) {
  q <- length(s$mag)
  p <- theta[1]
  
  return(gdual.mul(gdual.inv(gdual.add(gdual.mul(s, lsgdual.cdx(p - 1, q)),
                                       lsgdual.1dx(q))),
                   lsgdual.cdx(p, q)))
}

geometric2.lsgd <- function(s, theta) {
  q <- length(s$mag)
  p <- theta[1]
  
  return(gdual.mul(gdual.mul(gdual.inv(gdual.add(gdual.mul(s, lsgdual.cdx(p - 1, q)),
                                                 lsgdual.1dx(q))),
                             lsgdual.cdx(p, q)),
                   s))
}

### test code
y      <- 400*c(2, 5, 3)
lambda <- 400*c(20, 10, 5)
delta  <- c(0.3, 0.6)
rho    <- c(0.25, 0.25, 0.25)

runtime <- system.time({Alpha <- gdualforward(y, poisson.lsgd, lambda, bernoulli.lsgd, delta, rho, d = 0)})