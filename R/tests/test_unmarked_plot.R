#load("paramest_vs_lambda.RData")

library(ggplot2)
library(reshape)

require_complete_runs <- TRUE

x.axis = 'rho'
x.log = TRUE
file.prefix = '_newplots'

labels <- c()
cols   <- c()
if(do.pco) {
  labels <- c(labels, 'pco')
  cols   <- c(cols,   'red')
}
if(do.rgdfwd) {
  labels <- c(labels, 'rgdfwd')
  cols   <- c(cols,   'green')
}
if(do.apgffwd) {
  labels <- c(labels, 'apgffwd')
  cols   <- c(cols,   'blue')
}

# subset all records to the last run with any non-NA runtimes
if(require_complete_runs) {
  if(do.pco) {
    last_good_pco     <- max(which(apply(runtime_unmarked, 1, function(x) all(!is.na(x)))))
  } else { last_good_pco <- 0 }
  if(do.rgdfwd) {
    last_good_rgdfwd  <- max(which(apply(runtime_rgdfwd,   1, function(x) all(!is.na(x)))))
  } else { last_good_rgdfwd  <- 0 }
  if(do.apgffwd) {
    last_good_apgffwd <- max(which(apply(runtime_apgffwd,  1, function(x) all(!is.na(x)))))
  } else { last_good_apgffwd <- 0 }
} else {
  if(do.pco) {
    last_good_pco     <- max(which(apply(runtime_unmarked, 1, function(x) any(!is.na(x)))))
  } else { last_good_pco <- 0 }
  if(do.rgdfwd) {
    last_good_rgdfwd  <- max(which(apply(runtime_rgdfwd,   1, function(x) any(!is.na(x)))))
  } else { last_good_rgdfwd  <- 0 }
  if(do.apgffwd) {
    last_good_apgffwd <- max(which(apply(runtime_apgffwd,  1, function(x) any(!is.na(x)))))
  } else { last_good_apgffwd <- 0 }
}

last_good_run <- max(c(last_good_pco, last_good_rgdfwd, last_good_apgffwd))

test_vals <- test_vals[1:last_good_run]
runtime_unmarked <- runtime_unmarked[1:last_good_run,]
runtime_rgdfwd   <- runtime_rgdfwd[1:last_good_run,]
runtime_apgffwd  <- runtime_apgffwd[1:last_good_run,]
lambda_record_pco <- lambda_record_pco[1:last_good_run,]
iota_record_pco   <- iota_record_pco[1:last_good_run,]
gamma_record_pco  <- gamma_record_pco[1:last_good_run,]
omega_record_pco  <- omega_record_pco[1:last_good_run,]
det_record_pco    <- det_record_pco[1:last_good_run,]
lambda_record_rgdual <- lambda_record_rgdual[1:last_good_run,]
iota_record_rgdual   <- iota_record_rgdual[1:last_good_run,]
gamma_record_rgdual  <- gamma_record_rgdual[1:last_good_run,]
omega_record_rgdual  <- omega_record_rgdual[1:last_good_run,]
det_record_rgdual    <- det_record_rgdual[1:last_good_run,]
lambda_record_apgffwd <- lambda_record_apgffwd[1:last_good_run,]
iota_record_apgffwd   <- iota_record_apgffwd[1:last_good_run,]
gamma_record_apgffwd  <- gamma_record_apgffwd[1:last_good_run,]
omega_record_apgffwd  <- omega_record_apgffwd[1:last_good_run,]
det_record_apgffwd    <- det_record_apgffwd[1:last_good_run,]
lambda_record_gen <- lambda_record_gen[1:last_good_run]
iota_record_gen   <- iota_record_gen[1:last_good_run]
gamma_record_gen  <- gamma_record_gen[1:last_good_run]
omega_record_gen  <- omega_record_gen[1:last_good_run]
p_record_gen      <- p_record_gen[1:last_good_run]

labels <- c()
cols   <- c()
if(do.pco) {
  labels <- c(labels, 'unmarked')
  cols   <- c(cols,   'red')
}
if(do.rgdfwd) {
  labels <- c(labels, 'rgdfwd')
  cols   <- c(cols,   'green')
}
if(do.apgffwd) {
  labels <- c(labels, 'apgffwd')
  cols   <- c(cols,   'blue')
}

# plot runtimes
runtime.mean.df <- data.frame(x = test_vals)
if(do.pco) {
  runtime.mean.pco    <- rowMeans(runtime_unmarked, na.rm = TRUE)
  runtime.mean.df$pco <- runtime.mean.pco
}
if(do.rgdfwd) {
  runtime.mean.rgdfwd    <- rowMeans(runtime_rgdfwd, na.rm = TRUE)
  runtime.mean.df$rgdfwd <- runtime.mean.rgdfwd
}
if(do.apgffwd) {
  runtime.mean.apgffwd    <- rowMeans(runtime_apgffwd, na.rm = TRUE)
  runtime.mean.df$apgffwd <- runtime.mean.apgffwd
}
runtime.mean.df <- melt(runtime.mean.df, id=c("x"))

pdf(paste0(file.prefix, '_runtime.pdf'))
g <- ggplot(runtime.mean.df, aes(x=x)) +
  labs(title = paste0("Mean runtime (", n.reps, " trials)"),
       x     = x.axis,
       y     = "Mean runtime (s)") +
  theme(legend.title=element_blank()) +
  scale_color_manual(labels=labels, values = cols) +
  geom_line(aes(y = value, col = variable))
if(x.log)
  g <- g + scale_x_log10()
# plot(g)
print(g)
dev.off()

# plot lambda results
lambda.rmse.df <- data.frame(x = test_vals)
lambda.mae.df  <- data.frame(x = test_vals)
if(do.pco) {
  lambda.rmse.pco    <- sqrt(rowMeans((lambda_record_pco - replicate(n.reps, lambda_record_gen)) ^ 2, na.rm = TRUE))
  lambda.rmse.df$pco <- lambda.rmse.pco
  lambda.mae.pco    <- rowMeans(abs(lambda_record_pco - replicate(n.reps, lambda_record_gen)), na.rm = TRUE)
  lambda.mae.df$pco <- lambda.mae.pco
}
if(do.rgdfwd) {
  lambda.rmse.rgdfwd    <- sqrt(rowMeans((lambda_record_rgdfwd - replicate(n.reps, lambda_record_gen)) ^ 2, na.rm = TRUE))
  lambda.rmse.df$rgdfwd <- lambda.rmse.rgdfwd
  lambda.mae.rgdfwd    <- rowMeans(abs(lambda_record_rgdfwd - replicate(n.reps, lambda_record_gen)), na.rm = TRUE)
  lambda.mae.df$rgdfwd <- lambda.mae.rgdfwd
}
if(do.apgffwd) {
  lambda.rmse.apgffwd    <- sqrt(rowMeans((lambda_record_apgffwd - replicate(n.reps, lambda_record_gen)) ^ 2, na.rm = TRUE))
  lambda.rmse.df$apgffwd <- lambda.rmse.apgffwd
  lambda.mae.apgffwd    <- rowMeans(abs(lambda_record_apgffwd - replicate(n.reps, lambda_record_gen)), na.rm = TRUE)
  lambda.mae.df$apgffwd <- lambda.mae.apgffwd
}
lambda.rmse.df <- melt(lambda.rmse.df, id=c("x"))
lambda.mae.df  <- melt(lambda.mae.df, id=c("x"))

pdf(paste0(file.prefix, '_lambda_rmse.pdf'))
g <- ggplot(lambda.rmse.df, aes(x=x)) +
  labs(title = paste0("RMSE of lambda (", n.reps, " trials)"),
       x     = x.axis,
       y     = "RMSE") +
  theme(legend.title=element_blank()) +
  scale_color_manual(labels=labels, values = cols) +
  geom_line(aes(y = value, col = variable))
if(x.log)
  g <- g + scale_x_log10()
# plot(g)
print(g)
dev.off()

pdf(paste0(file.prefix, '_lambda_mae.pdf'))
g <- ggplot(lambda.mae.df, aes(x=x)) +
  labs(title = paste0("MAE of lambda (", n.reps, " trials)"),
       x     = x.axis,
       y     = "MAE") +
  theme(legend.title=element_blank()) +
  scale_color_manual(labels=labels, values = cols) +
  geom_line(aes(y = value, col = variable))
if(x.log)
  g <- g + scale_x_log10()
# plot(g)
print(g)
dev.off()

# plot gamma results
gamma.rmse.df <- data.frame(x = test_vals)
gamma.mae.df  <- data.frame(x = test_vals)
if(do.pco) {
  gamma.rmse.pco    <- sqrt(rowMeans((gamma_record_pco - replicate(n.reps, gamma_record_gen)) ^ 2, na.rm = TRUE))
  gamma.rmse.df$pco <- gamma.rmse.pco
  gamma.mae.pco    <- rowMeans(abs(gamma_record_pco - replicate(n.reps, gamma_record_gen)), na.rm = TRUE)
  gamma.mae.df$pco <- gamma.mae.pco
}
if(do.rgdfwd) {
  gamma.rmse.rgdfwd    <- sqrt(rowMeans((gamma_record_rgdfwd - replicate(n.reps, gamma_record_gen)) ^ 2, na.rm = TRUE))
  gamma.rmse.df$rgdfwd <- gamma.rmse.rgdfwd
  gamma.mae.rgdfwd    <- rowMeans(abs(gamma_record_rgdfwd - replicate(n.reps, gamma_record_gen)), na.rm = TRUE)
  gamma.mae.df$rgdfwd <- gamma.mae.rgdfwd
}
if(do.apgffwd) {
  gamma.rmse.apgffwd    <- sqrt(rowMeans((gamma_record_apgffwd - replicate(n.reps, gamma_record_gen)) ^ 2, na.rm = TRUE))
  gamma.rmse.df$apgffwd <- gamma.rmse.apgffwd
  gamma.mae.apgffwd    <- rowMeans(abs(gamma_record_apgffwd - replicate(n.reps, gamma_record_gen)), na.rm = TRUE)
  gamma.mae.df$apgffwd <- gamma.mae.apgffwd
}
gamma.rmse.df <- melt(gamma.rmse.df, id=c("x"))
gamma.mae.df  <- melt(gamma.mae.df, id=c("x"))

pdf(paste0(file.prefix, '_gamma_rmse.pdf'))
g <- ggplot(gamma.rmse.df, aes(x=x)) +
  labs(title = paste0("RMSE of gamma (", n.reps, " trials)"),
       x     = x.axis,
       y     = "RMSE") +
  theme(legend.title=element_blank()) +
  scale_color_manual(labels=labels, values = cols) +
  geom_line(aes(y = value, col = variable))
if(x.log)
  g <- g + scale_x_log10()
# plot(g)
print(g)
dev.off()

pdf(paste0(file.prefix, '_gamma_mae.pdf'))
g <- ggplot(gamma.mae.df, aes(x=x)) +
  labs(title = paste0("MAE of gamma (", n.reps, " trials)"),
       x     = x.axis,
       y     = "MAE") +
  theme(legend.title=element_blank()) +
  scale_color_manual(labels=labels, values = cols) +
  geom_line(aes(y = value, col = variable))
if(x.log)
  g <- g + scale_x_log10()
# plot(g)
print(g)
dev.off()

# plot omega results
omega.rmse.df <- data.frame(x = test_vals)
omega.mae.df  <- data.frame(x = test_vals)
if(do.pco) {
  omega.rmse.pco    <- sqrt(rowMeans((omega_record_pco - replicate(n.reps, omega_record_gen)) ^ 2, na.rm = TRUE))
  omega.rmse.df$pco <- omega.rmse.pco
  omega.mae.pco    <- rowMeans(abs(omega_record_pco - replicate(n.reps, omega_record_gen)), na.rm = TRUE)
  omega.mae.df$pco <- omega.mae.pco
}
if(do.rgdfwd) {
  omega.rmse.rgdfwd    <- sqrt(rowMeans((omega_record_rgdfwd - replicate(n.reps, omega_record_gen)) ^ 2, na.rm = TRUE))
  omega.rmse.df$rgdfwd <- omega.rmse.rgdfwd
  omega.mae.rgdfwd    <- rowMeans(abs(omega_record_rgdfwd - replicate(n.reps, omega_record_gen)), na.rm = TRUE)
  omega.mae.df$rgdfwd <- omega.mae.rgdfwd
}
if(do.apgffwd) {
  omega.rmse.apgffwd    <- sqrt(rowMeans((omega_record_apgffwd - replicate(n.reps, omega_record_gen)) ^ 2, na.rm = TRUE))
  omega.rmse.df$apgffwd <- omega.rmse.apgffwd
  omega.mae.apgffwd    <- rowMeans(abs(omega_record_apgffwd - replicate(n.reps, omega_record_gen)), na.rm = TRUE)
  omega.mae.df$apgffwd <- omega.mae.apgffwd
}
omega.rmse.df <- melt(omega.rmse.df, id=c("x"))
omega.mae.df  <- melt(omega.mae.df, id=c("x"))

pdf(paste0(file.prefix, '_omega_rmse.pdf'))
g <- ggplot(omega.rmse.df, aes(x=x)) +
  labs(title = paste0("RMSE of omega (", n.reps, " trials)"),
       x     = x.axis,
       y     = "RMSE") +
  theme(legend.title=element_blank()) +
  scale_color_manual(labels=labels, values = cols) +
  geom_line(aes(y = value, col = variable))
if(x.log)
  g <- g + scale_x_log10()
# plot(g)
print(g)
dev.off()

pdf(paste0(file.prefix, '_omega_mae.pdf'))
g <- ggplot(omega.mae.df, aes(x=x)) +
  labs(title = paste0("MAE of omega (", n.reps, " trials)"),
       x     = x.axis,
       y     = "MAE") +
  theme(legend.title=element_blank()) +
  scale_color_manual(labels=labels, values = cols) +
  geom_line(aes(y = value, col = variable))
if(x.log)
  g <- g + scale_x_log10()
# plot(g)
print(g)
dev.off()

# plot rho results
det.rmse.df <- data.frame(x = test_vals)
det.mae.df  <- data.frame(x = test_vals)
if(do.pco) {
  det.rmse.pco    <- sqrt(rowMeans((det_record_pco - replicate(n.reps, p_record_gen)) ^ 2, na.rm = TRUE))
  det.rmse.df$pco <- det.rmse.pco
  det.mae.pco    <- rowMeans(abs(det_record_pco - replicate(n.reps, p_record_gen)), na.rm = TRUE)
  det.mae.df$pco <- det.mae.pco
}
if(do.rgdfwd) {
  det.rmse.rgdfwd    <- sqrt(rowMeans((det_record_rgdfwd - replicate(n.reps, p_record_gen)) ^ 2, na.rm = TRUE))
  det.rmse.df$rgdfwd <- det.rmse.rgdfwd
  det.mae.rgdfwd    <- rowMeans(abs(det_record_rgdfwd - replicate(n.reps, p_record_gen)), na.rm = TRUE)
  det.mae.df$rgdfwd <- det.mae.rgdfwd
}
if(do.apgffwd) {
  det.rmse.apgffwd    <- sqrt(rowMeans((det_record_apgffwd - replicate(n.reps, p_record_gen)) ^ 2, na.rm = TRUE))
  det.rmse.df$apgffwd <- det.rmse.apgffwd
  det.mae.apgffwd    <- rowMeans(abs(det_record_apgffwd - replicate(n.reps, p_record_gen)), na.rm = TRUE)
  det.mae.df$apgffwd <- det.mae.apgffwd
}
det.rmse.df <- melt(det.rmse.df, id=c("x"))
det.mae.df  <- melt(det.mae.df, id=c("x"))

pdf(paste0(file.prefix, '_rho_rmse.pdf'))
g <- ggplot(det.rmse.df, aes(x=x)) +
  labs(title = paste0("RMSE of rho (", n.reps, " trials)"),
       x     = x.axis,
       y     = "RMSE") +
  theme(legend.title=element_blank()) +
  scale_color_manual(labels=labels, values = cols) +
  geom_line(aes(y = value, col = variable))
if(x.log)
  g <- g + scale_x_log10()
# plot(g)
print(g)
dev.off()

pdf(paste0(file.prefix, '_rho_mae.pdf'))
g <- ggplot(det.mae.df, aes(x=x)) +
  labs(title = paste0("MAE of rho (", n.reps, " trials)"),
       x     = x.axis,
       y     = "MAE") +
  theme(legend.title=element_blank()) +
  scale_color_manual(labels=labels, values = cols) +
  geom_line(aes(y = value, col = variable))
if(x.log)
  g <- g + scale_x_log10()
# plot(g)
print(g)
dev.off()

# plot iota results
iota.rmse.df <- data.frame(x = test_vals)
iota.mae.df  <- data.frame(x = test_vals)
if(do.pco) {
  iota.rmse.pco    <- sqrt(rowMeans((iota_record_pco - replicate(n.reps, iota_record_gen)) ^ 2, na.rm = TRUE))
  iota.rmse.df$pco <- iota.rmse.pco
  iota.mae.pco    <- rowMeans(abs(iota_record_pco - replicate(n.reps, iota_record_gen)), na.rm = TRUE)
  iota.mae.df$pco <- iota.mae.pco
}
if(do.rgdfwd) {
  iota.rmse.rgdfwd    <- sqrt(rowMeans((iota_record_rgdfwd - replicate(n.reps, iota_record_gen)) ^ 2, na.rm = TRUE))
  iota.rmse.df$rgdfwd <- iota.rmse.rgdfwd
  iota.mae.rgdfwd    <- rowMeans(abs(iota_record_rgdfwd - replicate(n.reps, iota_record_gen)), na.rm = TRUE)
  iota.mae.df$rgdfwd <- iota.mae.rgdfwd
}
if(do.apgffwd) {
  iota.rmse.apgffwd    <- sqrt(rowMeans((iota_record_apgffwd - replicate(n.reps, iota_record_gen)) ^ 2, na.rm = TRUE))
  iota.rmse.df$apgffwd <- iota.rmse.apgffwd
  iota.mae.apgffwd    <- rowMeans(abs(iota_record_apgffwd - replicate(n.reps, iota_record_gen)), na.rm = TRUE)
  iota.mae.df$apgffwd <- iota.mae.apgffwd
}
iota.rmse.df <- melt(iota.rmse.df, id=c("x"))
iota.mae.df  <- melt(iota.mae.df, id=c("x"))

pdf(paste0(file.prefix, '_iota_rmse.pdf'))
g <- ggplot(iota.rmse.df, aes(x=x)) +
  labs(title = paste0("RMSE of iota (", n.reps, " trials)"),
       x     = x.axis,
       y     = "RMSE") +
  theme(legend.title=element_blank()) +
  scale_color_manual(labels=labels, values = cols) +
  geom_line(aes(y = value, col = variable))
if(x.log)
  g <- g + scale_x_log10()
# plot(g)
print(g)
dev.off()

pdf(paste0(file.prefix, '_iota_mae.pdf'))
g <- ggplot(iota.mae.df, aes(x=x)) +
  labs(title = paste0("MAE of iota (", n.reps, " trials)"),
       x     = x.axis,
       y     = "MAE") +
  theme(legend.title=element_blank()) +
  scale_color_manual(labels=labels, values = cols) +
  geom_line(aes(y = value, col = variable))
if(x.log)
  g <- g + scale_x_log10()
# plot(g)
print(g)
dev.off()

# compute exact nll
if(do.pco && do.apgffwd) {
  source("nll_exact_eval.R")
  
  exact.nll.df <- data.frame(pco     = nll.exact.pco    [1:(n.experiments*n.reps)],
                             apgffwd = nll.exact.apgffwd[1:(n.experiments*n.reps)])
  
  pdf(paste0(file.prefix, '_exact_nll.pdf'))
  g <- ggplot(exact.nll.df, aes(x=pco, y=apgffwd)) +
    labs(title = paste0("Exact NLL at final optimization parameters"),
         x     = "PCO",
         y     = "APGFFWD") +
    geom_point() +
    geom_abline(intercept = 0, slope = 1)
  # plot(g)
  print(g)
  dev.off()
}