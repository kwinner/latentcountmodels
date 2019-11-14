clean <- TRUE

if(clean) {
  vars <- ls(all.names = TRUE)
  vars <- vars[vars != "clean"]
  vars <- vars[vars != "eval.mags"]
  vars <- vars[vars != "nll.apgf"]
  vars <- vars[vars != "nll.fwd"]
  vars <- vars[vars != "nll.truncpco"]
  vars <- vars[vars != "rt.apgf"]
  vars <- vars[vars != "rt.fwd"]
  vars <- vars[vars != "rt.truncpco"]
  vars <- vars[vars != "y_record"]
}

source("pcountOpen_rgdual.R")
source("pcountOpen_constrained_apgffwd.R")
# source("pcountOpen")

library(ggplot2)

inv.logit <- function(x) { exp(x) / (exp(x) + 1) }
logit <- function(x) { log(x) - log(1 - x) }

experiment.name <- "gen_y_eval3"

# M <- 2
# T <- 5
# lambda <- 15
# iota <- 10
# gamma <- 0.2
# # omega <- 0.8
# # omega <- test_vals[i.experiment]
# omega <- 0.4
# p <- 0.7
# y <- matrix(NA, M, T)
# N <- matrix(NA, M, T)
# S <- matrix(NA, M, T-1)
# O <- matrix(NA, M, T-1)
# I <- matrix(NA, M, T-1)
# N[,1] <- rpois(M, lambda)
# for(t in 1:(T-1)) {
#   S[,t] <- rbinom(M, N[,t], omega)
#   # S[,t] <- 0
#   O[,t] <- rpois(M, N[,t] * gamma)
#   I[,t] <- rpois(M, iota)
#   N[,t+1] <- S[,t] + O[,t] + I[,t]
# }
# y[] <- rbinom(M*T, N, p)

M <- 3
T <- 6
# lambda <- 12.3980227
# gamma  <- 0.3394449
# omega  <- 0.6567346
# p      <- 0.9518883
# iota   <- 1.0208007
lambda <- 80
gamma <- 0.95
omega <- 0.65
p <- 0.5
iota <- 8

n.reps <- 30

# y <- matrix(NA, M, T)
# N <- matrix(NA, M, T)
# S <- matrix(NA, M, T-1)
# O <- matrix(NA, M, T-1)
# I <- matrix(NA, M, T-1)
# N[,1] <- rpois(M, lambda)
# for(t in 1:(T-1)) {
#   S[,t] <- rbinom(M, N[,t], omega)
#   O[,t] <- rpois(M, N[,t] * gamma)
#   I[,t] <- rpois(M, iota)
#   N[,t+1] <- S[,t] + O[,t] + I[,t]
# }
# y[] <- rbinom(M*T, N, p)
# Y = sum(y)
# y_record[[(i.experiment - 1) * n.reps + i.rep]] <- y

# y <- matrix(c(13,12,11,19,17,16,19,16,17,17,18,21,10,16,17,12,20,18), nrow = 3, ncol = 6)
y <- matrix(c(35,31,27,24,39,34,32,31,33,28,32,30,25,36,25,23,43,32), nrow = 3, ncol = 6)

# Y = sum(y)

dynamics    = "trend"
immigration = TRUE

arrival.pgf <- pgf.poisson
# offspring.pgf   <- function(s, theta) {
#   return(pgf.bernoulli(s, theta['p']) * pgf.poisson(s, theta['lambda']))
# }
offspring.pgf <- pgf.poisson

theta.gen <- c(lambda, gamma, omega, p, iota)

# titles <- c("NLL vs lambda",
#             "NLL vs gamma",
#             "NLL vs omega",
#             "NLL vs rho",
#             "NLL vs iota",
#             "NLL vs lambda and iota",
#             "NLL vs Random direction 1",
#             "NLL vs Random direction 2",
#             "NLL vs Random direction 3",
#             "NLL vs Random direction 4",
#             "NLL vs Random direction 5",
#             "NLL vs Random direction 6",
#             "NLL vs Random direction 7",
#             "NLL vs Random direction 8",
#             "NLL vs Random direction 9",
#             "NLL vs Random direction 10")
titles <- c(
  "NLL vs rho",
  "NLL vs lambda",
  "NLL vs gamma",
  # "NLL vs delta",
  "NLL vs iota"
)
titles.rt <- c("Runtime vs rho",
               "Runtime vs lambda",
               "Runtime vs gamma",
               # "Runtime vs delta",
               "Runtime vs iota")

x_labs <- c(expression(rho), expression(lambda), expression(gamma), expression(iota))
# x_labs <- c("lambda", "gamma", "delta", "iota")

# titles <- c("NLL vs lambda")
# titles.rt <- c("Runtime vs lambda")
# x_labs <- c("lambda")

n.experiments <- length(titles)

if(clean) {
  eval.mags <- vector(n.experiments, mode = "list")
  nll.apgf <- vector(n.experiments, mode = "list")
  nll.fwd <- vector(n.experiments, mode = "list")
  nll.truncpco <- vector(n.experiments, mode = "list")
  rt.apgf <- vector(n.experiments, mode = "list")
  rt.fwd <- vector(n.experiments, mode = "list")
  rt.truncpco <- vector(n.experiments, mode = "list")
  y_record <- vector(n.experiments, mode = "list")
}

# x_labs <- c("lambda", "gamma", "delta", "rho", "iota")
# x_labs <- c("lambda", "gamma", "delta", "iota")

for(i.experiment in 1:n.experiments) {
  title <- titles[i.experiment]
  print(title)
  if(identical(title, "NLL vs lambda")) {
    eval.dir  <- c(1,0,0,0,0)
    theta.0   <- theta.gen
    theta.0[1] <- 0
    eval.mags[[i.experiment]] <- seq(-0.5, 0.5, 0.05) * lambda + lambda
  } else if(identical(title, "NLL vs gamma")) {
    eval.dir  <- c(0,1,0,0,0)
    theta.0   <- theta.gen
    theta.0[2] <- 0
    # eval.mags[[i.experiment]] <- seq(0, 1, 0.05)
    eval.mags[[i.experiment]] <- seq(-0.5, 0.25, 0.05) * gamma + gamma
  } else if(identical(title, "NLL vs delta")) {
    eval.dir  <- c(0,0,1,0,0)
    theta.0   <- theta.gen
    theta.0[3] <- 0
    eval.mags[[i.experiment]] <- seq(0, 1, 0.05)
  } else if(identical(title, "NLL vs rho")) {
    eval.dir  <- c(0,0,0,1,0)
    theta.0   <- theta.gen
    theta.0[4] <- 0
    eval.mags[[i.experiment]] <- seq(0.25, 0.75, 0.05)
  } else if(identical(title, "NLL vs iota")) {
    eval.dir  <- c(0,0,0,0,1)
    theta.0   <- theta.gen
    theta.0[5] <- 0
    eval.mags[[i.experiment]] <- seq(-0.75, 0.75, 0.1) * iota + iota
  } else if(identical(title, "NLL vs lambda and iota")) {
    eval.dir  <- c(1,0,0,0,1)
    theta.0   <- theta.gen
    eval.mags[[i.experiment]] <- seq(-0.5, 0.5, 0.05) * mean(lambda, iota) + mean(lambda, iota)
  } else if(grepl("Random", title)) {
    eval.dir  <- runif(5, min = -1 + 1e-6, max = 1-1e-6)
    eval.dir  <- eval.dir / sqrt(sum(eval.dir ^ 2))
    eval.dir  <- eval.dir * theta.gen
    
    theta.0   <- theta.gen
    eval.mags[[i.experiment]] <- seq(-0.5, 0.5, 0.05)
  } else {
    next()
  }
  
  # eval.param <- c("iota")
  # eval.vals <- seq(0.05, 0.95, 0.05)
  # if(length(eval.param) > 1) {
  #   title <- paste0("nll vs arrival params")
  # } else {
  #   title <- paste0("nll vs ", eval.param)
  # }
  
  # eval.params <- c(1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 30, 50, 100)
  # eval.params <- 0.5
  n.params <- length(eval.mags[[i.experiment]])
  nll.apgf[[i.experiment]] <- rep(0, n.params)
  nll.fwd[[i.experiment]]  <- rep(0, n.params)
  # nll.apgfpco <- rep(0, length(eval.mags))
  nll.truncpco[[i.experiment]] <- rep(0, n.params)
  rt.apgf[[i.experiment]] <- rep(0, n.params)
  rt.fwd[[i.experiment]]  <- rep(0, n.params)
  rt.truncpco[[i.experiment]] <- rep(0, n.params)
  y_record[[i.experiment]] <- vector(n.params * n.reps, mode="list")
  
  for(iter in 1:n.params) {
    mag.iter <- eval.mags[[i.experiment]][iter]
    print(paste0(iter,"/",n.params))
    
    theta.iter <- theta.0 + (mag.iter * eval.dir)
    
    lambda.eval <-     max(theta.iter[1], 1e-6)
    # gamma.eval  <- min(max(theta.iter[2], 1e-6), 1 - 1e-6)
    gamma.eval  <-     max(theta.iter[2], 1e-6)
    omega.eval  <- min(max(theta.iter[3], 1e-6), 1 - 1e-6)
    det.eval    <- min(max(theta.iter[4], 1e-6), 1 - 1e-6)
    iota.eval   <-     max(theta.iter[5], 1e-6)
    
    for(i.rep in 1:n.reps) {
      print(paste0('  rep ', i.rep, '/', n.reps))
      y <- matrix(NA, M, T)
      N <- matrix(NA, M, T)
      S <- matrix(NA, M, T-1)
      O <- matrix(NA, M, T-1)
      I <- matrix(NA, M, T-1)
      N[,1] <- rpois(M, lambda.eval)
      for(t in 1:(T-1)) {
        # S[,t] <- rbinom(M, N[,t], omega.eval)
        S[,t] <- 0
        O[,t] <- rpois(M, N[,t] * gamma.eval)
        I[,t] <- rpois(M, iota.eval)
        N[,t+1] <- S[,t] + O[,t] + I[,t]
      }
      y[] <- rbinom(M*T, N, det.eval)
      Y = sum(y)
      y_record[[i.experiment]][[(iter - 1) * n.reps + i.rep]] <- y
      
      umf <- unmarkedFramePCO(y = y, numPrimary=T)
      
      # restore original params for evaluation
      lamda.eval <- theta.gen[1]
      gamma.eval <- theta.gen[2]
      omega.eval <- theta.gen[3]
      det.eval   <- theta.gen[4]
      iota.eval  <- theta.gen[5]
      
      if(identical(dynamics, 'trend'))
        eval.at <- c(lambda.eval, gamma.eval, det.eval, iota.eval)
      else
        eval.at <- c(lambda.eval, gamma.eval, omega.eval, det.eval, iota.eval)
      
      print('    apgffwd')
      time.start <- proc.time()[3]
      nll.apgf[[i.experiment]][iter] <- nll.apgf[[i.experiment]][iter] + pcountOpen_apgffwd(~1, ~1, ~1, ~1, umf, immigration = immigration, dynamics = dynamics, eval.at = eval.at, nll.fun = "apgffwd", transform = FALSE) / n.reps
      rt.apgf[[i.experiment]][iter] <- rt.apgf[[i.experiment]][iter] + (proc.time()[3] - time.start) / n.reps
      print(paste0('    rt: ',(proc.time()[3] - time.start)))
      
      if(identical(dynamics, 'trend'))
        eval.at.trunc <- c(log(lambda.eval), log(gamma.eval), logit(det.eval), log(iota.eval))
      else
        eval.at.trunc <- c(log(lambda.eval), log(gamma.eval), logit(omega.eval), logit(det.eval), log(iota.eval))
      
      K <- K.default(y, det.eval)
      # K <- K.default(y, NULL)
      # K <- 100
      print(paste0('    trunc (K=', K, ')'))
      time.start <- proc.time()[3]
      nll.truncpco[[i.experiment]][iter] <- nll.truncpco[[i.experiment]][iter] + pcountOpen_apgffwd(~1, ~1, ~1, ~1, umf, immigration = immigration, dynamics = dynamics, eval.at = eval.at.trunc, nll.fun = "trunc", K = K, transform=TRUE) / n.reps
      rt.truncpco[[i.experiment]][iter] <- rt.truncpco[[i.experiment]][iter] + (proc.time()[3] - time.start) / n.reps
      print(paste0('    rt: ',(proc.time()[3] - time.start)))
      
      print('    gdfwd')
      time.start <- proc.time()[3]
      nll.fwd[[i.experiment]][iter] <- nll.fwd[[i.experiment]][iter] + pcountOpen_rgdual(~1, ~1, ~1, ~1, umf, immigration = immigration, dynamics = dynamics, eval.at = eval.at.trunc, nll.fun = "rgd") / n.reps
      rt.fwd[[i.experiment]][iter] <- rt.fwd[[i.experiment]][iter] + (proc.time()[3] - time.start) / n.reps
      print(paste0('    rt: ',(proc.time()[3] - time.start)))
    }
    # theta.arrival <- data.frame(lambda = c(lambda.eval, rep(iota.eval, T - 1)))
    # # theta.offspring <- data.frame(lambda = rep(gamma.eval, T - 1))
    # theta.offspring <- data.frame(p      = array(omega.eval, T - 1),
    #                               lambda = array(gamma.eval, T - 1))
    # rho <- data.frame(p = rep(det.eval, T))
    # # browser()
    # for(i.y in 1:nrow(y)) {
    #   time.start <- proc.time()[3]
    #   nll.apgf.iter <- apgffwd(y[i.y,], arrival.pgf, theta.arrival, offspring.pgf, theta.offspring, rho)
    #   rt.apgf[iter] <- rt.apgf[iter] + (proc.time()[3] - time.start)
    #   # browser()
    #   if(nll.apgf.iter$sign < 0)
    #     browser()
    #   nll.apgf[iter] <- nll.apgf[iter] - nll.apgf.iter$mag
    #   time.start <- proc.time()[3]
    #   nll.fwd[iter] <- nll.fwd[iter] - forward.ll(forward(y[i.y,], arrival.pgf, theta.arrival, offspring.pgf, theta.offspring, rho))
    #   rt.fwd[iter] <- rt.fwd[iter] + (proc.time()[3] - time.start)
    # }
  }
  
  # pdf(paste0(title, '.pdf'))
  # plot.df <- data.frame(x = eval.mags[[i.experiment]], apgf = nll.apgf[[i.experiment]], fwd = nll.fwd[[i.experiment]], truncpco = nll.truncpco[[i.experiment]])
  # g <- ggplot(plot.df, aes()) + 
  #   labs(x = x_labs[i.experiment], y = "NLL", title = title, colour = "Method") +
  #   theme(plot.title = element_text(hjust = 0.5)) +
  #   geom_point(aes(x = eval.mags[[i.experiment]], y = apgf, col = "apgffwd-s"), shape = 0) + 
  #   geom_point(aes(x = eval.mags[[i.experiment]], y = fwd,  col = "gdual-fwd"),   shape = 1) +
  #   geom_point(aes(x = eval.mags[[i.experiment]], y = truncpco,  col = "trunc"),   shape = 2) +
  #   guides(colour = guide_legend(override.aes = list(shape= c(0, 1, 2))))
  # # plot(g)
  # print(g)
  # dev.off()
  # 
  # pdf(paste0(title, '_rt', '.pdf'))
  # plot.df <- data.frame(x = eval.mags[[i.experiment]], apgf = rt.apgf[[i.experiment]], fwd = rt.fwd[[i.experiment]], truncpco = rt.truncpco[[i.experiment]])
  # g <- ggplot(plot.df, aes()) + 
  #   labs(x = x_labs[i.experiment], y = "runtime (s)", title = titles.rt[i.experiment], colour = "Method") +
  #   theme(plot.title = element_text(hjust = 0.5)) +
  #   geom_point(aes(x = eval.mags[[i.experiment]], y = apgf, col = "apgffwd-s"), shape = 0) + 
  #   geom_point(aes(x = eval.mags[[i.experiment]], y = fwd,  col = "gdual-fwd"),   shape = 1) +
  #   geom_point(aes(x = eval.mags[[i.experiment]], y = truncpco,  col = "trunc"),   shape = 2) +
  #   guides(colour = guide_legend(override.aes = list(shape= c(0, 1, 2))))
  # # plot(g)
  # print(g)
  # dev.off()
  # plot(eval.params, do.call(rbind, lapply(nll.apgf, as.numeric)), ylab = "nll", xlab = "eval.param")
}

save.image(file=paste0(experiment.name, ".RData"))