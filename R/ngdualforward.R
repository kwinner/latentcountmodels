source("ngdual.R")
source("generatingfunctions.R")

# y                          : vector of observed counts (len K)
# arrival.ngdual.liftedpgf   : function pointer for ngdual-lifted arrival pgf function
# theta.arrival              : list of arrival dist'n params (len K)
# offspring.ngdual.liftedpgf : function pointer for ngdual-lifted offspring pgf function
# theta.offspring            : list of offspring dist'n params (len K-1)
# theta.observ               : list of observ dist'n params (len K)
# d                          : length of final UTP, d >= 1
#                              i.e. final Alpha will have d-1 derivatives
#
# returns a list of ngduals for all K alpha messages
ngdualforward <- function(y,
                          arrival.ngdual.liftedpgf,
                          theta.arrival,
                          offspring.ngdual.liftedpgf,
                          theta.offspring,
                          theta.observ,
                          d=1) {
  
  K <- length(y)
  
  # storage for the alpha message utps
  Alpha <- vector(mode = "list", length = K)
  
  # browser()
  
  # recursive function for computing the Alpha messages
  # s_ds : value to evaluate A_k() at
  # k    : current index
  # q_k  : requested length of A_k message
  #
  # return <alpha_k, ds_k>_{q_k}
  liftA <- function(s,
                    k,
                    q_k) {
    if(k < 1) {
      # return a "dummy" utp for <1, dx>_{q_k}
      return(ngdual.1_dx(q_k))
    }
    
    u_du <- ngdual.x_dx(s * (1 - theta.observ[[k]]), q_k + y[k])
    
    # if(u_du$utp[1] <= 0)
    #   browser()
    
    # recursively get ngdual for incoming message
    if(k > 1) {
      F <- offspring.ngdual.liftedpgf(u_du, theta.offspring[[k-1]])
      
      # if(F$utp[1] <= 0)
      #   browser()
      
      s_prev <- exp(F$logZ) * F$utp[1]
      
      # if(s_prev <= 0)
        # browser()
      
      alpha_prev <- liftA(s_prev, k - 1, q_k + y[k])
      
      beta <- ngdual.compose(alpha_prev, F)
    } else {
      # no offspring for first message
      beta <- ngdual.1_dx(q_k + y[k])
    }
    
    # if(beta$utp[1] <= 0)
      # browser()
    
    G <- arrival.ngdual.liftedpgf(u_du, theta.arrival[[k]])
    
    # if(G$utp[1] <= 0)
    if(any(is.nan(G$utp)))
      browser()
    
    beta <- ngdual.mul(beta, G)
    
    # if(beta$utp[1] <= 0)
      # browser()
    
    alpha <- ngdual.deriv(beta, y[k])
    
    # if(alpha$utp[1] <= 0)
      # browser()
    
    s_ds <- ngdual.x_dx(s, q_k)
    
    # if(s_ds$utp[1] <= 0)
      # browser()
    
    alpha <- ngdual.affinecompose(alpha, 
                                  ngdual.scalar_mul(s_ds, 
                                                    1 - theta.observ[[k]]))
    
    # if(alpha$utp[1] <= 0)
      # browser()
    
    alpha <- ngdual.mul(alpha,
                        ngdual.pow(ngdual.scalar_mul(s_ds,
                                                     theta.observ[[k]]),
                                   y[k]))
    
    # if(alpha$utp[1] <= 0)
      # browser()
    
    alpha <- ngdual.scalar_mul_log(alpha, -lgamma(y[k] + 1))
    
    # if(alpha$utp[1] <= 0)
      # browser()
    
    Alpha[[k]] <<- alpha
    
    return(alpha)
  }
  
  liftA(1.0, K, d)
  
  return(Alpha)
}

ngdualforward.ll <- function(alpha) {
  if(is.nan(alpha$logZ) | is.nan(alpha$utp[1])) {
    return(-Inf)
  } else if(alpha$utp[1] < 0) {
    return(NaN)
  } else {
    return(alpha$logZ + log(alpha$utp[1]))
  }
}

# y <- as.numeric(list(2,9,12,14,9))
# K <- length(y)
# 
# theta.arrival   <- 50 * as.numeric(list(0.0257, 0.1163, 0.2104, 0.1504, 0.0428))
# theta.offspring <- rep(list(.5272), K-1)
# theta.observ    <- rep(list(0.5), K)
# arrival.pgf     <- poisson.ngdual
# offspring.pgf   <- bernoulli.ngdual
# 
# Alpha <- ngdualforward(y, arrival.pgf, theta.arrival, offspring.pgf, theta.offspring, theta.observ, d=1)