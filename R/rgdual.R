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

as.ls.numeric <- function(x) {
  # convert a linear system scalar/vector to logsign system
  x <- as.vector(x) # enforce dimensionality (this converts matrices/arrays to vectors)
  
  obj <- data.frame(mag  = log(abs(x)),
                    sign = as.integer(sign(x)))
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
  if(!is.ls(a)) { a <- as.ls(a) }
  if(!is.ls(b)) { b <- as.ls(b) }
  return(.Call("ls_add_R", a, b, PACKAGE=LIB_GDUAL))
}

'-.ls' <- function(a, b) {
  if(!is.ls(a)) { a <- as.ls(a) }
  if(!is.ls(b)) { b <- as.ls(b) }
  return(.Call("ls_sub_R", a, b, PACKAGE=LIB_GDUAL))
}

'*.ls' <- function(a, b) {
  if(!is.ls(a)) { a <- as.ls(a) }
  if(!is.ls(b)) { b <- as.ls(b) }
  return(.Call("ls_mul_R", a, b, PACKAGE=LIB_GDUAL))
}

'/.ls' <- function(a, b) {
  if(!is.ls(a)) { a <- as.ls(a) }
  if(!is.ls(b)) { b <- as.ls(b) }
  return(.Call("ls_div_R", a, b, PACKAGE=LIB_GDUAL))
}

'^.ls' <- function(a, r) {
  if(!is.ls(a)) { a <- as.ls(a) }
  return(.Call("ls_pow_R", a, r, PACKAGE=LIB_GDUAL))
}

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

'+.lsgd' <- function(a, b) {
  if(!is.lsgd(a)) { a <- lsgd.cdx(a, nrow(b)) }
  if(!is.lsgd(b)) { b <- lsgd.cdx(b, nrow(a)) }

  #assert length(a) == length(b)

  return(.Call("lsgd_add_R", a, b, PACKAGE=LIB_GDUAL))
}

'-.lsgd' <- function(a, b) {
  if(!is.lsgd(a)) { a <- lsgd.cdx(a, nrow(b)) }
  if(!is.lsgd(b)) { b <- lsgd.cdx(b, nrow(a)) }

  #assert length(a) == length(b)

  return(.Call("lsgd_sub_R", a, b, PACKAGE=LIB_GDUAL))
}

'*.lsgd' <- function(a, b) {
  if(!is.lsgd(a)) { a <- lsgd.cdx(a, nrow(b)) }
  if(!is.lsgd(b)) { b <- lsgd.cdx(b, nrow(a)) }

  #assert length(a) == length(b)

  return(.Call("lsgd_mul_R", a, b, PACKAGE=LIB_GDUAL))
}

'/.lsgd' <- function(a, b) {
  if(!is.lsgd(a)) { a <- lsgd.cdx(a, nrow(b)) }
  if(!is.lsgd(b)) { b <- lsgd.cdx(b, nrow(a)) }

  #assert length(a) == length(b)

  return(.Call("lsgd_div_R", a, b, PACKAGE=LIB_GDUAL))
}

'^.lsgd' <- function(a, r) {
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
  if(is.lsgd(x)) {
    q <- nrow(x)
    return(compose.lsgd(deriv.lsgd(f(lsgd.xdx(x, q + k)), 
                                   k), 
                        x))
  } else {
    return(deriv.lsgd(f(lsgd.xdx(x, k)), 
                      k))
  }
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
                    d = 1) {
  K <- length(y)
  
  Alpha <- vector(mode="list", length=K)
  
  liftGamma <- function(u, k) {
    if(k <= 1) {
      if(is.lsgd(u))
        q <- nrow(u)
      else
        q <- 1
      return(lsgd.1dx(q) * arrival.pgf(u, theta.arrival[[k]]))
    }
  }
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
  n      <- theta[1]
  p      <- theta[2]
  return(((1 - p) + (p * s)) ^ n)
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

t1 <- as.ls(c(1,2,3,4))
t2 <- as.ls(c(5,6,7,8))
t3 <- ls.zeros(4)

# t1 <- as.ls(1)
# t2 <- as.ls(8)
# t3 <- ls.zeros(1)
# t4 <- ls.zeros(1)
# t5 <- ls.zeros(1)

invisible(.Call("_ls_add_R", t3, t1, t2, PACKAGE=LIB_GDUAL))
# invisible(.Call("_ls_neg_R", t4, t3, PACKAGE=LIB_GDUAL))
# invisible(.Call("_ls_pow_R", t5, t4, 2, PACKAGE=LIB_GDUAL))

lsgd1 <- lsgd.xdx(5, 4)
invisible(lsgd2 <- .Call("lsgd_log_R", lsgd1))
