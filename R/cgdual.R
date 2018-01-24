dyn.load("../c/libgdual.so")

real2ls <- function(x) {
  return(ls(log(abs(x)), sign(x)))
}

ls <- function(mag, sign) {
  return(list(mag = mag, sign = sign))
}

is.ls <- function(x) {
  return(   typeof(x) == "list" 
         && names(x)  == c("mag", "sign"))
}

ls2real <- function(x.ls) {
  return(x.ls$sign * exp(x.ls$mag))
}

ls.is_zero <- function(x) {
  return(x$mag == -Inf || x$sign == 0)
}

ls.add <- function(x, y) {
  if(ls.is_zero(x))
    return(y)
  if(ls.is_zero(y))
    return(x)
  
  z <- real2ls(0)
  if(x$sign == y$sign)
    temp.sign <- 1
  else
    temp.sign <- -1

  if(x$mag > y$mag) {
    z$sign <- x$sign
    z$mag  <- x$mag + log1p(temp.sign * exp(y$mag - x$mag))
  } else {
    z$sign <- y$sign
    z$mag  <- y$mag + log1p(temp.sign * exp(x$mag - y$mag))
  }
  
  return(z)
}

ls.mul <- function(x, y) {
  return(ls(mag = x$mag + y$mag, sign = x$sign * y$sign))
}

ls.pow <- function(x, k) {
  return(ls(mag = k * x$mag, sign = sign(x$sign ^ k)))
}

lsgdual.1dx <- function(q) {
  return(real2ls(c(1, rep(0, q - 1))))
}

lsgdual.cdx <- function(k, q) {
  return(real2ls(c(k, rep(0, q - 1))))
}

lsgdual.cdx.ls <- function(k.ls, q) {
  # extend k.ls to order q by filling with zeros
  
  k.ls$mag  <- c(k.ls$mag,  rep(-Inf, q - 1))
  k.ls$sign <- c(k.ls$sign, rep(0,    q - 1))
  
  return(k.ls)
}

lsgdual.xdx <- function(x, q) {
  if(q == 1) {
    return(real2ls(x)) 
  } else {
    return(real2ls(c(x, 1, rep(0, q - 2))))
  }
}

lsgdual.xdx.ls <- function(k.ls, q) {
  # extend k.ls to order q by filling with zeros, but set second term to 1
  
  if(q == 1) {
    return(k.ls)
  } else {
    k.ls$mag  <- c(k.ls$mag,  0, rep(-Inf, q - 2))
    k.ls$sign <- c(k.ls$sign, 1, rep(0,    q - 2))
  }
  
  return(k.ls)
}

lsgdual.copy <- function(x, q) {
  x.q <- length(x$mag)
  if(missing(q) || q == x.q) {
    return(x) # r scoping handles the copy already
  } else if(q > x.q) {
    # extend x with zeros
    x$mag  <- c(x$mag,  rep(-Inf, q - x.q))
    x$sign <- c(x$sign, rep(0,    q - x.q))
    
    return(x)
  } else { #q < x.q
    # truncate x
    x$mag  <- x$mag[1:q]
    x$sign <- x$sign[1:q]
    
    return(x)
  }
}

gdual.unary_op <- function(fun.name, u) {
  n <- length(u$mag)
  
  result <- .C(fun.name, 
               v_mag  = double(n),
               v_sign = integer(n),
               u_mag  = as.double(u$mag),
               u_sign = as.integer(u$sign),
               nin    = as.integer(n),
               NAOK = TRUE)
  
  v <- list(mag=result$v_mag, 
           sign=result$v_sign)
  
  return(v)
}

gdual.binary_op <- function(fun.name, u, w) {
  n <- length(u$mag)
  
  result <- .C(fun.name, 
               v_mag  = double(n),
               v_sign = integer(n),
               u_mag  = as.double(u$mag),
               u_sign = as.integer(u$sign),
               w_mag  = as.double(w$mag),
               w_sign = as.integer(w$sign),
               nin    = as.integer(n),
               NAOK = TRUE)
  
  v <- list(mag=result$v_mag, 
            sign=result$v_sign)
  
  return(v)
}

gdual.scalar_op <- function(fun.name, u, k) {
  n <- length(u$mag)
  
  result <- .C(fun.name, 
               v_mag  = double(n),
               v_sign = integer(n),
               u_mag  = as.double(u$mag),
               u_sign = as.integer(u$sign),
               k      = as.double(k),
               nin    = as.integer(n),
               NAOK = TRUE)
  
  v <- list(mag=result$v_mag, 
            sign=result$v_sign)
  
  return(v)
}

gdual.add <- function(u, w) {
  return(gdual.binary_op("_gdual_add", u, w))
}

gdual.neg <- function(u) {
  return(gdual.unary_op("_gdual_neg", u))
}

gdual.exp <- function(u) {
  return(gdual.unary_op("_gdual_exp", u))
}

gdual.log <- function(u) {
  return(gdual.unary_op("_gdual_log", u))
}

gdual.pow <- function(u, r) {
  return(gdual.scalar_op("_gdual_pow", u, r))
}

gdual.inv <- function(u) {
  return(gdual.unary_op("_gdual_inv", u))
}

gdual.mul <- function(u, w) {
  return(gdual.binary_op("_gdual_mul", u, w))
}

gdual.div <- function(u, w) {
  return(gdual.binary_op("_gdual_div", u, w))
}

gdual.compose <- function(u, w) {
  u.mag0.cache  <- u$mag[1]
  u.sign0.cache <- u$sign[1]
  w.mag0.cache  <- w$mag[1]
  w.sign0.cache <- w$sign[1]
  
  # set first terms of u, w to zero
  u$mag[1]  <- -Inf
  u$sign[1] <- 0
  w$mag[1]  <- -Inf
  w$sign[1] <- 0
  
  # Horner's method truncated to q
  q <- length(u$mag)
  
  v <- lsgdual.cdx(0, q)
  v$mag[1]  <- u$mag[q]
  v$sign[1] <- u$sign[q]
  
  for(i in seq(q - 1, 1, by = -1)) {
    res <- gdual.mul(v, w)
    v$mag  <- res$mag[1:q]
    v$sign <- res$sign[1:q]
    
    res <- ls.add(ls(v$mag[1], v$sign[1]), ls(u$mag[i], u$sign[i]))
    v$mag[1]  <- res$mag[1]
    v$sign[1] <- res$sign[1]
  }
  
  # restore the cached value of u[1] to v[1]
  v$mag[1]  <- u.mag0.cache
  v$sign[1] <- u.sign0.cache
  
  return(v)
}

gdual.compose_affine <- function(u, w) {
  if(length(w$mag) <= 1) {
    return(u)
  } else {
    q <- length(u$mag)
    
    return(ls.mul(u, ls.pow(ls(w$mag[2], w$sign[2]), seq(0, q - 1))))
  }
}

gdual.deriv <- function(u, k) {
  q <- length(u$mag)
  
  # drop the first k terms
  u$mag  <- u$mag[-(1:k)]
  u$sign <- u$sign[-(1:k)]
  
  u$mag <- u$mag + lfallingfactorial(k, seq(k, q - 1))
  
  return(u)
}

gdual.diff <- function(f, x, k) {
  #Compute kth derivative of f evaluated at x
  #
  #f : lifted function
  #x : input
  #k : number of times to differentiate
  
  if(is.ls(x)) {
    #assume x is an lsgd. length 1 or 2 lsgds are special cased below
    #extend x to compute k extra utp terms, then pass to f
    q <- length(x$mag)

    result <- f(lsgdual.xdx.ls(ls(mag=x$mag[1], sign=x$sign[1]), q + k))    
    # if(q == 1)
    #   result <- f(lsgdual.xdx.ls(x, q + k))
    # else
    #   result <- f(lsgdual.copy(x, q + k))
    result.deriv <- gdual.deriv(result, k)
    
    if(q == 2)
      result.deriv <- gdual.compose_affine(result.deriv, x)
    else if(q > 2)
      result.deriv <- gdual.compose(result.deriv, x)
    
    return(result.deriv)
  } else { #!is.ls(x)
    #wrap x in an lsgd, then proceed as above,
    #  (though no compose will be needed since x is a constant)
    x <- lsgdual.xdx(x, k + 1)
    result <- f(x)
    result.deriv <- gdual.deriv(result, k)
    
    return(result.deriv)
  }
}


# UTILITY FUNCTIONS

# compute the Pochhammer symbol, (x)_n, in log space
lpoch <- function(x, n) {
  return(lgamma(x + n) - lgamma(x))
}

# compute the falling factorial in log space
lfallingfactorial <- function(k, i) {
  return(lpoch(i - k + 1, k))
}