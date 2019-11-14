# compute the nll using the exact method for all results from test_unmarked
source("rgdual.R")

nll.exact.pco     <- matrix(NA, n.experiments, n.reps)
nll.exact.apgffwd <- matrix(NA, n.experiments, n.reps)

arrival.pgf <- pgf.poisson

if(dynamics %in% c("constant")) {
  offspring.pgf   <- pgf.bernoulli
} else if(dynamics %in% c("autoreg")) {
  offspring.pgf   <- function(s, theta) {
    return(pgf.bernoulli(s, theta['p']) * pgf.poisson(s, theta['lambda']))
  }
} else if(dynamics %in% c("trend")) {
  offspring.pgf   <- pgf.poisson
}

for(i.experiment in 1:n.experiments) {
  for(i.rep in 1:n.reps) {
    print(paste0("Experiment #", i.experiment, "/", n.experiments, ", rep #", i.rep, "/", n.reps))
    
    y.iter <- y_record[[(i.experiment - 1) * n.reps + i.rep]]
    
    # pco
    lambda.pco <- lambda_record_pco[i.experiment, i.rep]
    det.pco    <- det_record_pco   [i.experiment, i.rep]
    gamma.pco  <- gamma_record_pco [i.experiment, i.rep]
    omega.pco  <- omega_record_pco [i.experiment, i.rep]
    iota.pco   <- iota_record_pco  [i.experiment, i.rep]
    if(is.finite(lambda.pco) &&
       is.finite(det.pco) &&
       is.finite(gamma.pco) &&
       is.finite(omega.pco) &&
       is.finite(iota.pco)) {
      
      # construct arrival dist'n
      if(dynamics %in% c("constant", "notrend"))
        theta.arrival <- data.frame(lambda = c(lambda.pco, array(gamma.pco, T - 1)))
      else if(immigration)
        theta.arrival <- data.frame(lambda = c(lambda.pco, array(iota.pco, T - 1)))
      else
        theta.arrival <- data.frame(lambda = c(lambda.pco, array(0, T - 1)))
      
      # construct offspring dist'n
      if(dynamics %in% c("constant"))
        theta.offspring <- data.frame(p = array(omega.pco, T - 1))
      else if(dynamics %in% c("autoreg"))
        theta.offspring <- data.frame(p      = array(omega.pco, T - 1),
                                      lambda = array(gamma.pco, T - 1))
      else if(dynamics %in% c("trend"))
        theta.offspring <- data.frame(lambda = array(gamma.pco, T - 1))
      
      theta.observ <- data.frame(p = array(det.pco, T))
      
      nll.exact.pco[i.experiment, i.rep] <- 0
      for(i.site in 1:M) {
        A <- forward(y.iter[i.site,],
                     arrival.pgf,
                     theta.arrival,
                     offspring.pgf,
                     theta.offspring,
                     theta.observ,
                     d = 0)
        nll.exact.pco[i.experiment, i.rep] <- nll.exact.pco[i.experiment, i.rep] - forward.ll(A)
      }
      
      print(paste0("      pco: ", nll.exact.pco[i.experiment, i.rep]))
    }
    
    # apgffwd
    lambda.apgffwd <- lambda_record_apgffwd[i.experiment, i.rep]
    det.apgffwd    <- det_record_apgffwd   [i.experiment, i.rep]
    gamma.apgffwd  <- gamma_record_apgffwd [i.experiment, i.rep]
    omega.apgffwd  <- omega_record_apgffwd [i.experiment, i.rep]
    iota.apgffwd   <- iota_record_apgffwd  [i.experiment, i.rep]
    if(is.finite(lambda.apgffwd) &&
       is.finite(det.apgffwd) &&
       is.finite(gamma.apgffwd) &&
       is.finite(omega.apgffwd) &&
       is.finite(iota.apgffwd)) {
      
      # construct arrival dist'n
      if(dynamics %in% c("constant", "notrend"))
        theta.arrival <- data.frame(lambda = c(lambda.apgffwd, array(gamma.apgffwd, T - 1)))
      else if(immigration)
        theta.arrival <- data.frame(lambda = c(lambda.apgffwd, array(iota.apgffwd, T - 1)))
      else
        theta.arrival <- data.frame(lambda = c(lambda.apgffwd, array(0, T - 1)))
      
      # construct offspring dist'n
      if(dynamics %in% c("constant"))
        theta.offspring <- data.frame(p = array(omega.apgffwd, T - 1))
      else if(dynamics %in% c("autoreg"))
        theta.offspring <- data.frame(p      = array(omega.apgffwd, T - 1),
                                      lambda = array(gamma.apgffwd, T - 1))
      else if(dynamics %in% c("trend"))
        theta.offspring <- data.frame(lambda = array(gamma.apgffwd, T - 1))
      
      theta.observ <- data.frame(p = array(det.apgffwd, T))
      
      nll.exact.apgffwd[i.experiment, i.rep] <- 0
      for(i.site in 1:M) {
        A <- forward(y.iter[i.site,],
                     arrival.pgf,
                     theta.arrival,
                     offspring.pgf,
                     theta.offspring,
                     theta.observ,
                     d = 0)
        nll.exact.apgffwd[i.experiment, i.rep] <- nll.exact.apgffwd[i.experiment, i.rep] - forward.ll(A)
      }
      
      print(paste0("  apgffwd: ", nll.exact.apgffwd[i.experiment, i.rep]))
    }
  }
}