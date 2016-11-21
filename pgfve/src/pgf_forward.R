PGFForwardAlgorithm <- function(lambda, delta, rho, y) {
	K <- length(y)

  messages <- list()

  #initial conditions
  a <- 0
  b <- 0
  f <- c(1)

  for (k in 1:K) {
    #arrivals.result = [a, b]
    arrivals.result <- ARRIVALS(a      = a, 
                                b      = b,
                                lambda = lambda[k])
    a <- arrivals.result$a
    b <- arrivals.result$b

    #evidence.result = [a, f]
    evidence.result <- EVIDENCE(a   = a,
                                f   = f,
                                y   = y[k],
                                rho = rho[k])
    a <- evidence.result$a
    f <- evidence.result$f

    #normalize.result = [b, f]
    normalize.result <- NORMALIZE(b = b,
                                  f = f)
    b <- normalize.result$b
    f <- normalize.result$f

    #cache result
    messages[[k]] <- list('a' = a,
                          'b' = b,
                          'f' = f)

    if (k <= K - 1) {
      #survivors.result = [a, b, f]
      survivors.result <- SURVIVORS(a     = a,
                                    b     = b,
                                    f     = f,
                                    delta = delta[k])
      a <- survivors.result$a
      b <- survivors.result$b
      f <- survivors.result$f
    }
  }

  loglikelihood <- log(sum(f)) + a + b

  return(list('loglikelihood' = loglikelihood,
              'messages'      = messages))
}

ARRIVALS <- function(a, b, lambda) {
  a.prime <- a + lambda
  b.prime <- b - lambda

  return(list('a.prime' = a.prime, 
              'b.prime' = b.prime))
}

EVIDENCE <- function(a, f, y, rho) {
  a.prime <- a * (1 - rho)
  
  f.prime <- 0
  df <- f

  for (y.i in 0:min(y, length(f)-1)) {
    #compute new derivative
    if (y.i != 0) {
      df <- df[-1]  #discard first term (coefficient of constant term)
      df <- df * 1:length(df) #multiply coefficients by index
    }

    #compute denominator
    denom <- gamma(y.i + 1) * gamma(y - y.i + 1)

    #move (1-alpha) inside f to the coefficients
    numCo <- (1 - rho) ^ (0:(length(df) - 1))

    #multiply by a^(y - y.i)
    acPowers <- a ^ (y - y.i)
    
    result <- df * numCo / denom * acPowers

    #align the output and the derivative
    n.max <- max(length(f.prime), length(result))
    f.prime <- c(f.prime, rep(0, n.max - length(f.prime)))
    result  <- c(result,  rep(0, n.max - length(result)))

    #fold into output polynomial
    f.prime <- f.prime + result
  }

  #multiply all coefficients by rho ^ y
  f.prime <- f.prime * (rho ^ y)

  #multiply by s^y (by appending zeros to the front of f.prime)
  f.prime <- c(rep(0,y), f.prime)

  return(list('a.prime' = a.prime, 
              'f.prime' = f.prime))
}

SURVIVORS <- function(a, b, f, delta) {
  a.prime <- a * delta
  b.prime <- b + a * (1 - delta)

  f.prime <- COMPOSE_POLY_HORNER_LINEAR(f, c(1-delta, delta))

  return(list('a.prime' = a.prime,
              'b.prime' = b.prime,
              'f.prime' = f.prime))
}

COMPOSE_POLY_HORNER_LINEAR <- function(f, g) {
  n <- length(f)
  h <- f[n]

  for (i.f in seq(n-1,1,-1)) {
    h <- c(h * g[1], 0) +
         c(f[i.f],   h * g[2])
  }

  return(h)
}

COMPOSE_POLY_HORNER <- function(f, g) {
  n <- length(f)
  h <- f[n]

  for (i.f in seq(n-1,1,-1)) {
    h <- convolve(h, rev(g), type = "o")
    h[1] <- h[1] + f[i.f]
  }

  return(h)
}

#factor out the largest coefficient of f and move it into exp(b)
NORMALIZE <- function(b, f) {
  Z <- max(f)
  f.prime <- f / Z
  b.prime <- b + log(Z)

  return(list('b.prime' = b.prime,
              'f.prime' = f.prime))
}