LIB_GDUAL = "lib_gdual"
dyn.load(paste(LIB_GDUAL,".so", sep=''))

################################################################################
# A note on the LS and LSGD representations below:
# Both objects are fundamentally just a 2 column data frame, with columns:
#   'mag'  := real
#   'sign' := integer
# with each row containing one term of the vector (or coefficient of the lsgd)
################################################################################

##################################################################
#                           LS METHODS                           #
##################################################################

# empty vector in lns format
ls.empty <- function() {
  obj <- data.frame(mag  = double(),
                    sign = integer())
  class(obj) <- c('ls', 'data.frame')
  return(obj)
}

# vector of zeros in lns format
ls.zeros <- function(q) {
  obj <- data.frame(mag  = rep.int(-Inf, q), # vector of -Inf
                    sign = integer(q))       # vector of zeros
  class(obj) <- c('ls', 'data.frame')
  return(obj)
}

# vector of ones in lns format
ls.ones <- function(q) {
  obj <- data.frame(mag  = double(q),        # vector of zeros
                    sign = rep.int(1L, q))   # vector of (integer) ones
  class(obj) <- c('ls', 'data.frame')
  return(obj)
}

# test if an object conforms to the ls format
#   if strict is FALSE, then only the class is checked
#   if strict is TRUE, then the structure of the object is checked as well
is.ls <- function(x, strict=FALSE) {
  return('ls' %in% class(x) && 
          (!strict || 
            ('mag'  %in% names(x) &&
             'sign' %in% names(x) &&
             typeof(x$mag)  == 'double' &&
             typeof(x$sign) == 'integer')
        ))
}

# convert a number (or vector of) in standard linear system to a LS object
as.ls <- function(x) { UseMethod("as.ls", x) }
as.ls.ls <- function(x) { return(x) }
as.ls.data.frame <- function(x) { return(as.ls.numeric(as.numeric(x))) }

as.ls.numeric <- function(x) {
  # convert a linear system scalar/vector to logsign system
  x <- as.vector(x) # enforce dimensionality (this converts matrices/arrays to vectors)
  
  obj <- data.frame(mag  = log(abs(x)),
                    sign = as.integer(sign(x)))
  class(obj) <- c('ls', 'data.frame')
  return(obj)
}

as.ls.magsign <- function(mag, sign=1L) {
  # convert a magnitude and sign to a proper ls object
  mag  <- as.vector(mag) # enforce dimensionality (this converts matrices/arrays to vectors)
  sign <- as.vector(sign)
  
  obj <- data.frame(mag  = as.double(mag),
                    sign = as.integer(sign))
  class(obj) <- c('ls', 'data.frame')
  return(obj)
}

# convert a number (or vector of) in logsign system to a standard double vector
# note: this same generic method also provides as.numeric.ls(...)
as.double.ls <- function(x) {
  if(is.double(x))
    return(x)
  else if(is.ls(x, strict=TRUE)) {
    return(x$sign * exp(x$mag))
  } else
    # fail case
    callNextMethod(x)
}

# modified print function for ls objects with different default behavior and
# automatic addition of linear-space format
print.ls <- function (x, ..., digits = NULL, quote = FALSE, right = TRUE) 
{
  n <- nrow(x)
  if (n == 0L) {
    cat(gettext("[length 0 logsign number]"))
  } else {
    m <- as.matrix(format.data.frame(x, 
                                     digits = digits, 
                                     na.encode = FALSE))
    
    # append the double representation of x
    m <- cbind(m, as.double(x))
    colnames(m)[3] <- '[as.double]'
    
    dimnames(m)[[1L]] <- rep.int("", n)
    
    print(m, ..., quote = quote, right = right)
  }
  invisible(x)
}

'+.ls' <- function(a, b) {
  if(is.lsgd(a) || is.lsgd(b)) { return(add.lsgd(a,b)) }
  if(!is.ls(a)) { a <- as.ls(a) }
  if(!is.ls(b)) { b <- as.ls(b) }
  return(.Call("ls_add_R", a, b, PACKAGE=LIB_GDUAL))
}

'-.ls' <- function(a, b) {
  #unary negation
  if(missing(b)) {
    if(is.lsgd(a)) { return(neg.lsgd(a)) }
    else           { return(neg.ls(a)) }
  }
  if(is.lsgd(a) || is.lsgd(b)) { return(sub.lsgd(a,b)) }
  if(!is.ls(a)) { a <- as.ls(a) }
  if(!is.ls(b)) { b <- as.ls(b) }
  return(.Call("ls_sub_R", a, b, PACKAGE=LIB_GDUAL))
}

'*.ls' <- function(a, b) {
  if(is.lsgd(a) || is.lsgd(b)) { return(mul.lsgd(a,b)) }
  if(!is.ls(a)) { a <- as.ls(a) }
  if(!is.ls(b)) { b <- as.ls(b) }
  return(.Call("ls_mul_R", a, b, PACKAGE=LIB_GDUAL))
}

'/.ls' <- function(a, b) {
  if(is.lsgd(a) || is.lsgd(b)) { return(div.lsgd(a,b)) }
  if(!is.ls(a)) { a <- as.ls(a) }
  if(!is.ls(b)) { b <- as.ls(b) }
  return(.Call("ls_div_R", a, b, PACKAGE=LIB_GDUAL))
}

'^.ls' <- function(a, r) {
  # print(paste('hello ', class(r)))
  if(!is.ls(a)) { a <- as.ls(a) }
  return(.Call("ls_pow_R", a, r, PACKAGE=LIB_GDUAL))
}

'>.ls' <- function(a, b) {
  if(!is.ls(b) && is.na(b)) { return(NA) }
  if(!is.ls(b)) { b <- as.ls(b) }
  
  if(!is.finite(b)) {
    if(b$sign < 0 && !(a$mag == Inf && a$sign < 0))
      return(TRUE)
    else if(b$sign > 0 && !(a$mag == Inf && a$sign > 0))
      return(FALSE)
    else # other degenerate cases of b (NaN or comparing Inf w/ Inf or -Inf w/ -Inf)
      return(NA)
  }
  
  if(a$sign == b$sign) {
    if(a$sign == 0)
      return(FALSE)
    else if(a$sign > 0)
      return(a$mag > b$mag)
    else if(a$sign < 0)
      return(a$mag < b$mag)
  } else
    return(a$sign > b$sign)
}

'>=.ls' <- function(a, b) {
  if(!is.ls(b) && (is.na(b) || is.nan(b))) { return(NA) }
  if(!is.ls(b)) { b <- as.ls(b) }
  
  if(!is.finite(b)) {
    if(b$sign < 0 && !(a$mag == Inf && a$sign < 0))
      return(TRUE)
    else if(b$sign > 0 && !(a$mag == Inf && a$sign > 0))
      return(FALSE)
    else # other degenerate cases of b (NaN or comparing Inf w/ Inf or -Inf w/ -Inf)
      return(NA)
  }
  
  if(a$sign == b$sign) {
    if(a$sign == 0)
      return(TRUE)
    else if(a$sign > 0)
      return(a$mag >= b$mag)
    else if(a$sign < 0)
      return(a$mag <= b$mag)
  } else
    return(a$sign > b$sign)
}

'<.ls' <- function(a, b) {
  if(!is.ls(b) && is.na(b)) { return(NA) }
  if(!is.ls(b)) { b <- as.ls(b) }
  
  if(!is.finite(b)) {
    if(b$sign < 0 && !(a$mag == Inf && a$sign < 0))
      return(FALSE)
    else if(b$sign > 0 && !(a$mag == Inf && a$sign > 0))
      return(TRUE)
    else # other degenerate cases of b (NaN or comparing Inf w/ Inf or -Inf w/ -Inf)
      return(NA)
  }
  
  if(a$sign == b$sign) {
    if(a$sign == 0)
      return(FALSE)
    else if(a$sign > 0)
      return(a$mag < b$mag)
    else if(a$sign < 0)
      return(a$mag > b$mag)
  } else
    return(a$sign < b$sign)
}

'<=.ls' <- function(a, b) {
  if(!is.ls(b) && is.na(b)) { return(NA) }
  if(!is.ls(b)) { b <- as.ls(b) }
  
  if(!is.finite(b)) {
    if(b$sign < 0 && !(a$mag == Inf && a$sign < 0))
      return(FALSE)
    else if(b$sign > 0 && !(a$mag == Inf && a$sign > 0))
      return(TRUE)
    else # other degenerate cases of b (NaN or comparing Inf w/ Inf or -Inf w/ -Inf)
      return(NA)
  }
  
  if(a$sign == b$sign) {
    if(a$sign == 0)
      return(FALSE)
    else if(a$sign > 0)
      return(a$mag <= b$mag)
    else if(a$sign < 0)
      return(a$mag >= b$mag)
  } else
    return(a$sign < b$sign)
}

'==.ls' <- function(a, b) {
  if(!is.ls(b)) { b <- as.ls(b) }
  
  return(a$mag == b$mag && a$sign == b$sign)
}

'abs.ls' <- function(a) {
  if(a$sign < 0) {
    a$sign <- 1L
  }
  return(a)
}

'is.finite.ls' <- function(a) {
  return(is.finite(a$mag))
}

'is.infinite.ls' <- function(a) {
  return(is.infinite(a$mag))
}

'is.nan.ls' <- function(a) {
  return(is.nan(a$mag) || is.nan(a$sign))
}

# '[.ls' <- function(x, i, j, drop = FALSE) {
#   print(1)

#   NextMethod()
# }

# todo: needs dispatcher (`neg(obj)` doesn't work)
neg.ls <- function(a) {
  return(.Call("ls_neg_R", a,    PACKAGE=LIB_GDUAL))
}

# todo: needs dispatcher (`inv(obj)` doesn't work)
inv.ls <- function(a) {
  return(.Call("ls_inv_R", a,    PACKAGE=LIB_GDUAL))
}

exp.ls <- function(a) {
  return(.Call("ls_exp_R", a,    PACKAGE=LIB_GDUAL))
}

log.ls <- function(a) {
  return(.Call("ls_log_R", a,    PACKAGE=LIB_GDUAL))
}

##################################################################
#                          LSGD METHODS                          #
##################################################################

# object representing a dual number for <1, dx>_q in LNS (S3)
lsgd.1dx <- function(q) {
  if(q == 0)
    obj <- ls.empty()
  else
    obj <- rbind(ls.ones(1), ls.zeros(q - 1))
  class(obj) <- c('lsgd', 'ls', 'data.frame')
  return(obj)
}

# object representing a dual number for <c, dx>_q in LNS (S3)
lsgd.cdx <- function(c, q) {
  if(q == 0)
    obj <- ls.empty()
  else {
    if(is.ls(c))
      obj <- rbind(c[1,], ls.zeros(q - 1))
    else
      obj <- rbind(as.ls(c), ls.zeros(q - 1))
  }
  
  class(obj) <- c('lsgd', 'ls', 'data.frame')
  return(obj)
}

# object representing a dual number for <x, dx>_q in LNS (S3)
# the parameter x below is the current value of x
lsgd.xdx <- function(x, q) {
  if(q == 0)
    obj <- ls.empty()
  else if(q == 1) {
    if(is.ls(x))
      obj <- x[1,]
    else
      obj <- as.ls(x)
  } else {
    if(is.ls(x))
      obj <- rbind(x[1,], ls.ones(1), ls.zeros(q - 2))
    else
      obj <- rbind(as.ls(x), ls.ones(1), ls.zeros(q - 2))
  }
  class(obj) <- c('lsgd', 'ls', 'data.frame')
  return(obj)
}

# strip lsgd from an object to stop treating it as a gdual and instead as plain ls
as.ls.lsgd <- function(x) { 
  class(x) <- c('ls', 'data.frame')
  return(x)
}

# test if an object conforms to the lsgd format
#   if strict is FALSE, then only the class is checked
#   if strict is TRUE, then the structure of the object is checked as well
is.lsgd <- function(x, strict=FALSE) {
  return('lsgd' %in% class(x) && 
           (!strict || 
              is.ls(x)))
}

print.lsgd <- function (x, ..., digits = NULL, quote = FALSE, right = TRUE)
{
  n <- nrow(x)
  if (n == 0L) {
    cat(gettext("[length 0 lsgd]"))
  } else {
    m <- as.matrix(format.data.frame(x,
                                     digits = digits,
                                     na.encode = FALSE))

    # append the double representation of x
    m <- cbind(m, as.double(x))
    colnames(m)[3] <- '[as.double]'

    dimnames(m)[[1L]] <- rep.int("", n)

    print(m, ..., quote = quote, right = right)
  }
  invisible(x)
}

# NOTE: binary operators for LSGDs are not overloaded
#       instead dispatch in '+.ls' dispatches to these if either
#       argument is an lsgd
add.lsgd <- function(a, b) {
  if(!is.lsgd(a)) { a <- lsgd.cdx(a, nrow(b)) }
  if(!is.lsgd(b)) { b <- lsgd.cdx(b, nrow(a)) }

  #assert length(a) == length(b)

  return(.Call("lsgd_add_R", a, b, PACKAGE=LIB_GDUAL))
}

sub.lsgd <- function(a, b) {
  if(!is.lsgd(a)) { a <- lsgd.cdx(a, nrow(b)) }
  if(!is.lsgd(b)) { b <- lsgd.cdx(b, nrow(a)) }

  #assert length(a) == length(b)

  return(.Call("lsgd_sub_R", a, b, PACKAGE=LIB_GDUAL))
}

mul.lsgd <- function(a, b) {
  if(!is.lsgd(a)) { a <- lsgd.cdx(a, nrow(b)) }
  if(!is.lsgd(b)) { b <- lsgd.cdx(b, nrow(a)) }

  #assert length(a) == length(b)

  return(.Call("lsgd_mul_R", a, b, PACKAGE=LIB_GDUAL))
}

div.lsgd <- function(a, b) {
  if(!is.lsgd(a)) { a <- lsgd.cdx(a, nrow(b)) }
  if(!is.lsgd(b)) { b <- lsgd.cdx(b, nrow(a)) }

  #assert length(a) == length(b)

  return(.Call("lsgd_div_R", a, b, PACKAGE=LIB_GDUAL))
}

'^.lsgd' <- function(a, r) {
  if (is.integer(r)) ( r <- as.double(r) )
  # print(paste('world ', class(r)))
  return(.Call("lsgd_pow_R", a, r, PACKAGE=LIB_GDUAL))
}

# todo: needs dispatcher (`neg(obj)` doesn't work)
neg.lsgd <- function(a) {
  return(.Call("lsgd_neg_R", a,    PACKAGE=LIB_GDUAL))
}

# todo: needs dispatcher (`inv(obj)` doesn't work)
inv.lsgd <- function(a) {
  return(.Call("lsgd_inv_R", a,    PACKAGE=LIB_GDUAL))
}

exp.lsgd <- function(a) {
  return(.Call("lsgd_exp_R", a,    PACKAGE=LIB_GDUAL))
}

log.lsgd <- function(a) {
  return(.Call("lsgd_log_R", a,    PACKAGE=LIB_GDUAL))
}

compose.lsgd <- function(a, b) {
  #assert is.lsgd(a) && is.lsgd(b)
  return(.Call("lsgd_compose_R", a, b, PACKAGE=LIB_GDUAL))
}

compose_affine.lsgd <- function(a, b) {
  #assert is.lsgd(a) && is.lsgd(b)
  return(.Call("lsgd_compose_affine_R", a, b, PACKAGE=LIB_GDUAL))
}

deriv.lsgd <- function(a, k) {
  q <- nrow(a)
  
  # drop the lowest order terms from a
  a <- a[(k+1):q,]
  
  # LS mult a by the logfallingfactorial
  a$mag <- a$mag + lfallingfactorial(k, seq(k, q - 1))
  
  return(a)
}

diff.lsgd <- function(f, x, k) {
  if(k == 0) {
    return(f(x))
  } else if(is.lsgd(x)) {
    q <- nrow(x)
    return(compose.lsgd(deriv(f(lsgd.xdx(x, q + k)), 
                              k), 
                        x))
  } else {
    return(deriv(f(lsgd.xdx(x, 1 + k)), 
                 k))
  }
}

'[.lsgd' <- function(x, i, j, drop = FALSE) {
  x <- NextMethod()
  class(x) <- c('ls', 'data.frame')
  return(x)
}

##################################################################
#                       FORWARD ALGORITHM                        #
##################################################################

forward <- function(y,
                    arrival.pgf,
                    theta.arrival,
                    branch.pgf,
                    theta.branch,
                    theta.observ,
                    d = 0) {
  K <- length(y)
  
  Alpha <- vector(mode="list", length=K)
  
  # wrap the arrival/branch pgfs, which may each be a list of pgfs
  # note: if arrival.pgf/branch.pgf are lists, they need to be length K or K - 1 accordingly
  if (is.list(arrival.pgf))
    arrival.pgf.select <- function(u, k) { arrival.pgf[k](u, theta.arrival[k,,drop=F]) }
  else
    arrival.pgf.select <- function(u, k) { arrival.pgf(u, theta.arrival[k,,drop=F]) }
  if (is.list(branch.pgf))
    branch.pgf.select <- function(u, k) { branch.pgf[k](u, theta.branch[k,,drop=F]) }
  else
    branch.pgf.select <- function(u, k) { branch.pgf(u, theta.branch[k,,drop=F]) }
  
  Gamma <- function(u, k) {
    # branch.pgf(u, theta.branch[k-1,,drop=F])
    return(A(branch.pgf.select(u, k-1), k-1) *
           arrival.pgf.select(u, k))
  }
  
  A <- function(s, k) {
    if(k < 1)
      return(1.0)
    
    if(is.lsgd(s))
      q <- nrow(s)
    else
      q <- 1
    
    Gamma_k <- function(u_k) { Gamma(u_k, k) }
    
    const <- (s ^ y[k]) * lsgd.cdx(as.ls.magsign(mag=(y[k] * log(theta.observ[k,]) - lgamma(y[k] + 1)),
                                                 sign=1L),
                                   q)
    alpha <- const * diff.lsgd(Gamma_k, s * (1 - theta.observ[k,]), y[k])
    Alpha[[k]] <<- alpha
    
    return(alpha)
  }
  
  diff.lsgd(function(s) A(s, K), 1.0, d)
  
  return(Alpha)
}

forward.ll <- function(Alpha) {
  return(Alpha[[length(Alpha)]]$mag)
}

##################################################################
#                              PGFS                              #
##################################################################

pgf.poisson    <- function(s, theta) {
  return(exp(theta$lambda * (s - 1)))
}

pgf.bernoulli   <- function(s, theta) {
  return((1 - theta$p) + (theta$p * s))
}

pgf.binomial    <- function(s, theta) {
  return(((1 - theta$p) + (theta$p * s)) ^ theta$n)
}

pgf.negbin      <- function(s, theta) {
  return((theta$p / (1 - ((1 - theta$p) * s))) ^ theta$r)
}

pgf.logarithmic <- function(s, theta) {
  return(log(1 - (theta$p * s)) / log(1 - theta$p))
}

pgf.geometric   <- function(s, theta) {
  return(theta$p / (1 - ((1 - theta$p) * s)))
}

pgf.geometric2  <- function(s, theta) {
  return((theta$p * s) / (1 - ((1 - theta$p) * s)))
}

# a special pgf for a Uniform(0,0) dist'n (for readability)
pgf.zero <- function(s, theta) {
  return(1)
}

# a special pgf for a Uniform(1,1) dist'n (for readability)
pgf.one <- function(s, theta) {
  return(s)
}

##################################################################
#                        UTILITY METHODS                         #
##################################################################

# compute the Pochhammer symbol, (x)_n, in log space
lpoch <- function(x, n) {
  return(lgamma(x + n) - lgamma(x))
}

# compute the falling factorial in log space
lfallingfactorial <- function(k, i) {
  return(lpoch(i - k + 1, k))
}

# t1 <- as.ls(c(1,2,3,4))
# t2 <- as.ls(c(5,6,7,8))
# t3 <- ls.zeros(4)
# 
# # t1 <- as.ls(1)
# # t2 <- as.ls(8)
# # t3 <- ls.zeros(1)
# # t4 <- ls.zeros(1)
# # t5 <- ls.zeros(1)
# 
# invisible(.Call("_ls_add_R", t3, t1, t2, PACKAGE=LIB_GDUAL))
# # invisible(.Call("_ls_neg_R", t4, t3, PACKAGE=LIB_GDUAL))
# # invisible(.Call("_ls_pow_R", t5, t4, 2, PACKAGE=LIB_GDUAL))
# 
# lsgd1 <- lsgd.xdx(5, 4)
# invisible(lsgd2 <- .Call("lsgd_log_R", lsgd1))
# 
# y      <- 300*c(1,2,3,1,3)
# lambda <- 300*data.frame(lambda = c(2.5, 6, 6, 6, 6))
# delta  <- data.frame(lambda = c(0.5, 0.5, 0.5, 0.5))
# rho    <- data.frame(p      = c(0.2, 0.2, 0.2, 0.2, 0.2))
# # # # y      <- c(2)
# # # # lambda <- data.frame(lambda=c(5))
# # # # delta  <- data.frame(p=c(0.3))
# # # # rho    <- data.frame(p=c(0.25))
# # # 
# rt <- 0
# for (i in 1:10) {
#   rt <- rt + system.time({A <- forward(y, pgf.poisson, lambda, pgf.poisson, delta, rho)})['elapsed']
# }
# print(forward.ll(A))
# print(rt)
# # 
# # print(A)
# print(forward.ll(A))
# # print(run_time)