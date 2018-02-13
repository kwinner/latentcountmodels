#todo:
# * eliminate "denormalization" operations wherever possible
# * parameterize q in all methods, particularly when it is possible for F, G to have different q

if(!require("matrixStats")) install.packages("matrixStats")

###################################################
# FACTORY METHODS FOR COMMON NGDUAL CONSTRUCTIONS
###################################################

#construct an ngdual for f(x) = 1
#inputs: q, the number of derivatives
ngdual.1_dx <- function(q) {
  utp <- numeric(q)
  utp[1] <- 1.0
  
  logZ <- 0.0
  
  return(list(logZ=logZ, utp=utp))
}

#construct an ngdual for a constant
#inputs: C, the constant
#        q, the number of derivatives
ngdual.C_dx <- function(C, q) {
  utp <- numeric(q)
  utp[1] <- 1.0

  logZ <- log(C)
  
  return(list(logZ=logZ, utp=utp))
}

#construct an ngdual for a variable with respect to itself
#inputs: x, the value of the variable and
#        q, the number of derivatives
ngdual.x_dx <- function(x, q) {
  # if x is an ngdual already, extract the denormalized value of x
  if(is.list(x)) {
    x <- exp(x$logZ) * x$utp[1]
  }
  
  utp <- numeric(q)
  
  if(x > 1.0) {
    utp[1] <- 1.0
    if(q > 1) {
      #\frac{d}{dx} x = 1.0
      utp[2] <- 1.0/x #division by x is normalizing
    }
    
    logZ <- log(x)
  } else {
    utp[1] <- x
    if(q > 1) {
      #\frac{d}{dx} x = 1.0
      utp[2] <- 1.0
    }
    
    logZ <- 0.0
  }
  
  return(list(logZ=logZ, utp=utp))
}

###################################################
# NGDUAL UTILITIES
###################################################

# build an ngdual from a utp
ngdual.normalize <- function(utp) {
  Z <- max(1, abs(utp))
  utp <- utp / Z
  
  return(list(logZ=log(Z), utp=utp))
}

# build an ngdual from a utp in logspace
ngdual.normalize.log <- function(logutp) {
  logZ <- max(logutp)
  logutp <- logutp - logZ
  
  return(list(logZ=logZ, utp=exp(logutp)))
}

# convert an ngdual to a utp
ngdual.denormalize <- function(ngd) {
  return(exp(ngd$logZ) * ngd$utp)
}

###################################################
# LOG SPACE UTILITIES
###################################################

# compute the Pochhammer symbol, (x)_n, in log space
lpoch <- function(x, n) {
  return(lgamma(x + n) - lgamma(x))
}

# compute the falling factorial in log space
lfallingfactorial <- function(k, i) {
  return(lpoch(i - k + 1, k))
}

###################################################
# CORE NGDUAL OPERATIONS
###################################################

# add a scalar term to F (normalization is nontrivial)
ngdual.scalar_add <- function(F, c) {
  out.utp <- F$utp
  
  # TODO: if F$utp[1] + (c / np.exp(F$logZ)) > 1, then renormalization can be handled explicitly
  out.utp[1] <- out.utp[1] + (c / exp(F$logZ))
  
  out <- ngdual.normalize(out.utp)
  
  # add normalization from F
  out$logZ <- out$logZ + F$logZ
  
  return(out)
}

# multiply F by a scalar term
ngdual.scalar_mul <- function(F, c) {
  if(c > 0) {
    # push c into logZ
    F$logZ <- F$logZ + log(c)
    return(F)
  } else if(c < 0) {
    # invert the utp, but use -c for the normalization
    F$logZ <- F$logZ + log(-c)
    F$utp  <- -1 * F$utp
    return(F)
  } else { #c = 0
    F$logZ <- 0
    F$utp  <- numeric(length(F$utp))
    return(F)
  }
}

# multiply F by a scalar term (in log space)
ngdual.scalar_mul_log <- function(F, logc) {
  F$logZ <- F$logZ + logc
  
  return(F)
}

# compute <F * g, dx>_q from <F, dx>_q and <g, dx>_q
ngdual.mul <- function(F, G) {
  q <- length(F$utp)
  
  out.utp <- convolve(F$utp, rev(G$utp), type="o")[1:q]
  
  # browser()
  
  # R's convolve has some stability issues when some entries are zero,
  # which we correct for here (when the true (F∘G)[i] is 0)
  nz.F <- tail(which(F$utp != 0), 1) # position of last non-zero coeff in F
  nz.G <- tail(which(G$utp != 0), 1) # ditto for G
  if(nz.F + nz.G <= q) {
    out.utp[(nz.F + nz.G):q] <- 0
  }
  
  # browser()
  
  # normalize out
  out <- ngdual.normalize(out.utp)
  
  # add normalization from F, G
  out$logZ = out$logZ + F$logZ + G$logZ
  
  return(out)
}

#compute <1/F, dx>_q from <F, dx>_q
# note: F will need to be "denormalized", may be unstable
ngdual.reciprocal <- function(F) {
  q       <- length(F$utp)
  out.utp <- numeric(q)
  
  # denormalize F
  F_star.utp <- ngdual.denormalize(F)
  
  # evaluate the first term
  out.utp[1] <- 1.0 / F_star.utp[1]
  
  # from algopy
  if(q > 1) {
    for(i in seq(1, q-1)) {
      out.utp[i+1] <- out.utp[1] * (-sum(rev(F_star.utp[seq(2,i+1)]) * out.utp[seq(1,i)]))
    }
  }
  
  # normalize out
  out <- ngdual.normalize(out.utp)
  
  return(out)
}

# compute <exp(F), dx>_q from <F, dx>_q
ngdual.exp <- function(F) {
  q       <- length(F$utp)
  out.utp <- numeric(q)
  
  # subtract off a correction term (similar to logsumexp stability trick)
  correction <- exp(F$logZ) * F$utp[1]
  F_prime <- ngdual.scalar_add(F, -correction)
  
  Z <- exp(F_prime$logZ)
  
  # evaluate the first term
  out.utp[1] <- exp(F_prime$utp[1]) ^ Z
  
  # subset the derivative terms of F, mul by index
  F_prime.utp_tilde <- F_prime$utp[2:q] * seq(1, q - 1)
   
  # from algopy
  if(q > 1) {
    for(i in seq(1, q-1)) {
      out.utp[i+1] = Z * sum(rev(out.utp[seq(1,i)]) * F_prime.utp_tilde[seq(1,i)]) / i
    }
  }
  
  # normalize out
  out <- ngdual.normalize(out.utp)
  
  # restore correction
  out$logZ <- out$logZ + correction
  
  return(out)
}



# # compute <exp(F), dx>_q from <F, dx>_q but improve stability by computing each term in logspace
# ngdual.exp.logspace <- function(F) {
#   q           <- length(F$utp)
#   log.out.utp <- numeric(q)
#   
#   # compute the first term of exp(f)
#   Z <- exp(F$logZ)
#   log.expf <- Z * F$utp[1] # cache this value (in logspace)
#   log.out.utp[1] <- 0 # essentially translate F to 0
#   
#   # construct \tilde{F}, which is essentially the derivative terms of F
#   F.utp.tilde <- F$utp[2:q] * seq(1, q - 1)
#   
#   for(i in seq(2,q)) {
#     # preslice the vectors for this iteration
#     F.iter <- F.utp.tilde[seq(i-1, 1, -1)]
#     log.out.utp.iter <- log.out.utp[seq(1,i-1)]
#     
#     # remove any entries where F is zero (will become zero in logsumexp anyways)
#     log.out.utp.iter <- log.out.utp.iter[F.iter != 0]
#     F.iter           <- F.iter[F.iter != 0]
#     
#     log.out.utp[i] <- F$logZ - log(i-1) + matrixStats::logSumExp(log.out.utp.iter + log(F.iter))
#   }
#   
#   # handle normalization (explicitly)
#   out.logZ <- max(0, log.out.utp)
#   log.out.utp <- log.out.utp - out.logZ
#   out.logZ <- out.logZ + log.expf
#   
#   # convert utp out of logspace and return
#   return(list(logZ = out.logZ, utp = exp(log.out.utp)))
# }
# # ngdual.exp <- ngdual.exp.logspace

# logsumexp <- function(X, X.sign = NA) {
#   # w/o sign argument or with all nonnegative signs, default to standard logSumExp
#   # would be nice to remove this dependence
#   if(is.na(X.sign) || all(X.sign >= 0)) {
#     result.mag <- matrixStats::logSumExp(X)
#     if(result.mag == -Inf) {
#       result.sign <- 0
#     } else {
#       result.sign <- 1
#     }
#     return(c(result.mag, result.sign))
#   } else {
#     X.mag  <- abs(X)
# 
#     Z.pos <- matrixStats::logSumExp(X.mag[X.sign == 1])
#     Z.neg <- matrixStats::logSumExp(X.mag[X.sign == -1])
# 
#     if(Z.pos >= Z.neg) {
#       result.sign <- 1
#       result.mag  <- log(1 - exp(Z.neg - Z.pos)) + Z.pos
#     } else {
#       result.sign <- -1
#       result.mag  <- log(1 - exp(Z.pos - Z.neg)) + Z.neg
#     }
# 
#     return(c(result.mag, result.sign))
#   }
# }

## ported from scipy.special.logsumexp
logsumexp <- function(a, b = NA) {
  a.max <- max(a)
  
  if(!is.finite(a.max)) {
    a.max <- 0
  }
  
  if(all(!is.na(b))) {
    tmp <- b * exp(a - a.max)
  } else {
    tmp <- exp(a - a.max)
  }
  
  s <- sum(tmp)
  
  sgn <- sign(s)
  out <- log(abs(s))
  
  return(c(out, sgn))
}

# # compute <exp(F), dx>_q from <F, dx>_q but improve stability by computing each term in logspace
# ngdual.exp.logspace <- function(F) {
#   q            <- length(F$utp)
#   log.out.utp  <- numeric(q)
#   sign.out.utp <- numeric(q)
#   
#   # compute the first term of exp(f)
#   Z <- exp(F$logZ)
#   log.expf <- Z * F$utp[1] # cache this value (in logspace)
#   log.out.utp[1] <- log.expf # essentially translate F to 0
#   sign.out.utp[1] <- 1
#   
#   # construct \tilde{F}, which is essentially the derivative terms of F
#   F.utp.tilde <- F$utp[2:q] * seq(1, q - 1)
#   log.F.utp.tilde  <- log(abs(F.utp.tilde)) # log magnitude of \tilde{F}
#   sign.F.utp.tilde <- sign(F.utp.tilde)     # sign of \tilde{F}
#   
#   for(i in seq(2,q)) {
#     # preslice the vectors for this iteration
#     log.F.iter        <- log.F.utp.tilde[seq(i-1, 1, -1)]
#     sign.F.iter       <- sign.F.utp.tilde[seq(i-1, 1, -1)]
#     log.out.utp.iter  <- log.out.utp[seq(1, i-1)]
#     sign.out.utp.iter <- sign.out.utp[seq(1, i-1)]
#     
#     # remove any entries where F is zero (will become zero in logsumexp anyways)
#     log.out.utp.iter  <- log.out.utp.iter[sign.F.iter != 0]
#     sign.out.utp.iter <- sign.out.utp.iter[sign.F.iter != 0]
#     log.F.iter        <- log.F.iter[sign.F.iter != 0]
#     sign.F.iter       <- sign.F.iter[sign.F.iter != 0]
#     
#     browser()
#     
#     logsumexp.res <- logsumexp(log.out.utp.iter + log.F.iter, sign.out.utp.iter * sign.F.iter)
#     log.out.utp[i] <- F$logZ - log(i-1) + logsumexp.res[1]
#     sign.out.utp[i] <- logsumexp.res[2]
#     
#     browser()
#   }
#   
#   # handle normalization (explicitly)
#   out.logZ <- max(0, log.out.utp)
#   log.out.utp <- log.out.utp - out.logZ
#   # out.logZ <- out.logZ + log.expf
#   
#   # convert utp out of logspace and return
#   return(list(logZ = out.logZ, utp = sign.out.utp * exp(log.out.utp)))
# }

ngdual.exp.logspace <- function(F) {
  q <- length(F$utp)
  
  # output utp in signlogspace
  H.utp.logabs <- numeric(q)
  H.utp.sign   <- numeric(q)
  
  # Ft = \tilde{F} = F[2:q] * (1:q-1)
  Ft.utp.logabs <- log(abs(F$utp[2:q])) + log(seq(1, q-1))
  Ft.utp.sign   <- sign(F$utp)
  
  # compute the first term of the output utp
  H.utp.logabs[1] <- exp(F$logZ) * F$utp[1]
  H.utp.sign[1]   <- 1
  
  for(i in seq(1,q-1)) {
    # slice the vectors for this for loop iteration
    Ft.utp.logabs.iter <- Ft.utp.logabs[seq(1, i)]
    Ft.utp.sign.iter   <- Ft.utp.sign  [seq(1, i)]
    H.utp.logabs.iter  <- rev(H.utp.logabs[seq(1, i)])
    H.utp.sign.iter    <- rev(H.utp.sign  [seq(1, i)])
    
    # G = rev(H.iter) * Ft.iter
    G.logabs <- Ft.utp.logabs.iter + H.utp.logabs.iter
    G.sign   <- Ft.utp.sign.iter   * H.utp.sign.iter
    
    # logsumexp with sign
    res <- logsumexp(G.logabs, G.sign)
    H.utp.logabs[i+1] <- res[1] + F$logZ - log(i)
    H.utp.sign[i+1]   <- res[2]
  }
  
  out.logZ <- max(H.utp.logabs)
  log.out.utp <- H.utp.logabs - out.logZ
  
  return(list(logZ = out.logZ, utp = H.utp.sign * exp(log.out.utp)))
}

# compute <log(F), dx>_q from <F, dx>_q
ngdual.log <- function(F) {
  q       <- length(F$utp)
  out.utp <- numeric(q)
  
  # evaluate the first term
  out.utp[1] <- log(F$utp[1])
  
  # correct for normalization of F
  out.utp[1] <- out.utp[1] + F$logZ
  
  # from algopy
  if(q > 1) {
    for(i in seq(2, q)) {
      if(i == 2) {
        out.utp[i] <- (F$utp[i] * (i-1)) / F$utp[1]
      } else {
        out.utp[i] <- (F$utp[i] * (i-1) - sum(rev(F$utp[seq(2,i-1)]) * out.utp[seq(2,i-1)])) / F$utp[1]
      }
    }
    out.utp[2:q] <- out.utp[2:q] / seq(1, q-1)
  }
  
  # normalize out
  out <- ngdual.normalize(out.utp)
  
  return(out)
}

# compute <f^k, dx>_q from <f, dx>_q
# note: uses 3 ngdual operations, but should be more stable
#       f^k = exp(k * log(f))
ngdual.pow <- function(F, k) {
  return(ngdual.exp(ngdual.scalar_mul(ngdual.log(F), k)))
}

# compute <G, dx>_q from <G, dF>_q and <F, dx>_r
# intuitively computes <G(F), dx>_q
# note: F will need to be "denormalized", may be unstable
ngdual.compose <- function(G, F) {
  q       <- length(G$utp)
  out.utp <- numeric(q)
  
  # restore unnormalized F
  F_star.utp <- ngdual.denormalize(F)
  
  # cache first term of G, then clear first terms of F,G
  G.utp.1.cache <- G$utp[1]
  G$utp[1] <- 0
  F_star.utp[1] <- 0
  
  # Horner's method truncated to q
  out.utp[1] <- G$utp[q]
  for(i in seq(q - 1, 1, -1)) {
    out.utp <- convolve(out.utp, rev(F_star.utp), type="o")[1:q]
    out.utp[1] <- out.utp[1] + G$utp[i]
  }
  
  # restore cached values
  out.utp[1] <- G.utp.1.cache
  
  # normalize out
  out <- ngdual.normalize(out.utp)
  
  # add normalization from G
  out$logZ <- out$logZ + G$logZ
  
  return(out)
}

# compute <G, dx>_q from <G, dF>_q and <F, dx>_r
#  where r = 1 or 2
# intuitively computes <G(F), dx>_q
# note: F will need to be "denormalized", may be unstable
ngdual.affinecompose <- function(G, F) {
  q.F <- length(F$utp)
  if(q.F <= 1) {
    # composition with a constant or q == 1 F is basically a no-op
    return(G)
  } else {
    q       <- length(G$utp)
    out.utp <- G$utp
    
    # unnormalize F
    F_star.utp <- ngdual.denormalize(F)
    
    # no need for Horner's method, utp composition uses only the 2nd and higher
    # coefficients, of which F has only 1 nonzero in this case
    out.utp <- out.utp * (F_star.utp[2] ^ seq(0, q - 1))
    
    # normalize out
    out <- ngdual.normalize(out.utp)
    
    # add normalization from G
    out$logZ <- out$logZ + G$logZ
    
    return(out)
  }
}


# compute <d^k/dx^k F, dx>_{q-k} from <F, dx>_q
ngdual.deriv <- function(F, k) {
  q       <- length(F$utp)
  out.utp <- F$utp
  
  # compute the vector of falling factorial terms in log space
  factln <- lfallingfactorial(k, seq(k, q-1))
  
  # drop the lower order terms
  out.utp <- out.utp[seq(k+1, q)]
  
  # cache negative signs before switching to logspace
  out.utp.signs <- sign(out.utp)
  out.utp       <- log(abs(out.utp))
  
  # multiply out.utp and the factorials in logspace
  out.utp <- out.utp + factln
  
  # normalize out (in log space)
  out <- ngdual.normalize.log(out.utp)
  
  # add normalization from F
  out$logZ <- out$logZ + F$logZ
  
  # restore signs of utp terms
  out$utp <- out$utp * out.utp.signs
  
  return(out)
}

