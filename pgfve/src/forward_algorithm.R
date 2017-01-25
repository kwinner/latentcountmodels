ForwardAlgorithm <- function(y, lambda, delta, rho, N_max) {
  if (is.vector(y))
    return(ForwardAlgorithmOneSite(y, lambda, delta, rho, N_max))
}

ForwardAlgorithmOneSite <- function(y, lambda, delta, rho, N_max) {
  #compute dimensions
  K <- length(y) #num of primary sampling occasions

  #scale up param inputs

  #initialize intermediate terms
  psi   <- matrix(0, K, N_max + 1)
  gamma <- matrix(0, K, N_max + 1)
  alpha <- matrix(0, K, N_max + 1)

  #p(zero initial survivors = 1)
  psi[1, 1] <- 1

  for (k in 1:K) {
    #compute arrival distribution
    arrival_distn <- dpois(0:N_max, lambda)

    #convolve psi_k and arriv to get gamma, using R's unintuitive conv
    gamma[k,] <- convolve(psi[k,], rev(arrival_distn), type = "o")

    #compute observ distribution, alpha
    observ_distn <- dbinom(y[k], 0:N_max, rho)
    alpha[k,] <- gamma[k,] * observ_distn

    #compute the incoming message for the next layer
    if (k < K) {
      #r cannot do dbinom over a 2D matrix in a way that maintains readability
      for (j in 0:N_max) {
        psi[k+1,] <- sum(alpha[k,] * dbinom(j, 0:N_max, delta[k]))
      }
    }
  }

  return(alpha)
}