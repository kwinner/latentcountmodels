source("ngdual.R")

poisson.ngdual <- function(s, theta) {
  lambda <- theta[1]
  
  return(ngdual.exp(ngdual.scalar_mul(ngdual.scalar_add(s, -1.0), 
                                      lambda)))
}

bernoulli.ngdual <- function(s, theta) {
  p <- theta[1]
  
  return(ngdual.scalar_add(ngdual.scalar_mul(s, p),
                           1 - p))
}

binomial.ngdual <- function(s, theta) {
  n <- theta[1]
  p <- theta[2]
  
  return(ngdual.pow(ngdual.scalar_add(ngdual.scalar_mul(s, p),
                                      1 - p),
                    n))
}

negbin.ngdual <- function(s, theta) {
  r <- theta[1]
  p <- theta[2]
  
  return(ngdual.pow(ngdual.scalar_mul(ngdual.reciprocal(ngdual.scalar_add(ngdual.scalar_mul(s, p - 1),
                                                                          1)),
                                      p),
                    r))
}

logarithmic.ngdual <- function(s, theta) {
  p <- theta[1]
  
  return(ngdual.scalar_mul(ngdual.log(ngdual.scalar_add(ngdual.scalar_mul(s, -p),
                                                        1)),
                           1 / log(1 - p)))
}

geometric.ngdual <- function(s, theta) {
  p <- theta[1]
  
  return(ngdual.scalar_mul(ngdual.reciprocal(ngdual.scalar_add(ngdual.scalar_mul(s, p - 1),
                                                               1)),
                           p))
}

geometric2.ngdual <- function(s, theta) {
  p <- theta[1]
  
  return(ngdual.mul(ngdual.scalar_mul(ngdual.reciprocal(ngdual.scalar_add(ngdual.scalar_mul(s, p - 1),
                                                                          1)),
                                      p),
                    s))
}