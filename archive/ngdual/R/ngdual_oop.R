##########################################################
# SETUP THE NGDUAL CLASS AND OVERLOAD SOME UTILITY METHODS
##########################################################

setClass("ngdual", representation(logZ = "numeric", utp = "numeric"))

check_ngdual <- function(object) {
  errors <- character()

  if(!is.atomic(object@logZ)) {
    errors <- c(errors, "Z must be atomic.")
  }
  
  if(!is.finite(object@logZ)) {
    errors <- c(errors, "Z must be finite.")
  }
  
  if(!all(is.finite(object@utp))) {
    errors <- c(errors, "All entries of UTP must be finite.")
  }
  
  if(length(errors) == 0) TRUE else errors
}

is.ngdual <- function(object) {
  return(inherits(object, "ngdual"))
}

###################################################
# FACTORY METHODS FOR COMMON NGDUAL CONSTRUCTIONS
###################################################

#construct an ngdual for f(x) = 1
#inputs: q, the number of derivatives
ngdual.1_dx <- function(q) {
  utp <- numeric(q)
  utp[1] <- 1.0
  
  logZ <- 0.0
  
  return(new("ngdual", logZ=logZ, utp=utp))
}

#construct an ngdual for a constant
#inputs: C, the constant
#        q, the number of derivatives
ngdual.C_dx <- function(C, q) {
  utp <- numeric(q)
  utp[1] <- 1.0

  logZ <- log(C)
  
  return(new("ngdual", logZ=logZ, utp=utp))
}

#construct an ngdual for a variable with respect to itself
#inputs: x, the value of the variable and
#        q, the number of derivatives
ngdual.x_dx <- function(x, q) {
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
  
  return(new("ngdual", logZ=logZ, utp=utp))
}

###################################################
# OVERLOADED SIMPLE/ACCESSOR FUNCTIONS
###################################################

setMethod("length", signature("ngdual"),
  function(x) {
    return(length(x@utp))
  }
)

setMethod("[", signature(x = "ngdual", i = "numeric", j = "missing"),
  function(x, i, j, ..., drop=TRUE) {
    return(exp(x@logZ) * x@utp[i])
  }
)

###################################################
# CORE NGDUAL OPERATIONS
###################################################
setGeneric("compose", function(F, G) {
  standardGeneric("compose")
})
setMethod("compose", signature(F = "ngdual", G = "ngdual"),
  function(F, G) {
    "COMPOSE"
  }
)

setGeneric("deriv", function(F, k) {
  standardGeneric("deriv")
})
setMethod("deriv", signature(F = "ngdual", k = "numeric"),
          function(F, k) {
            "COMPOSE"
          }
)