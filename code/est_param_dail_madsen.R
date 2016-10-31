library(rjags)
library(dplyr)
library(tidyr)
library(lazyeval)
library(ggplot2)

K <- 10
n_sites <- 10
params <- c('lambda', 'delta', 'rho')

# Create simulation data
lambda <- 10                 # rate of new arrivals
delta <- 0.5                 # survival probability
rho <- 0.8                   # detection probability

true_params <- c(lambda, delta, rho)
names(true_params) <- params

m <- matrix(rpois(n_sites * K, lambda), nrow = n_sites)
n <- matrix(rep(NA, n_sites * K), nrow = n_sites)
z <- matrix(rep(NA, n_sites * (K-1)), nrow = n_sites)

n[, 1] <- m[, 1]
for (k in 2:K) {
  z[, k - 1] <- rbinom(n_sites, n[k - 1], delta)
  n[, k] <- m[, k] + z[, k - 1]
}
y <- matrix(rbinom(n_sites * K, n, rho), nrow = n_sites)

# Initial values
#z0 <- y[1:K-1]
m0 <- y

# Run JAGS
n_iter <- 10000000
n_chains <- 2
model <- jags.model('dail_madsen_unk_param.bug',
                    data = list('y' = y, 'K' = K, 'S' = n_sites),
                    inits = list('m' = m0), # 'z' = z0
                    n.chains = n_chains,
                    n.adapt = 0)

t <- system.time(
  samples <- coda.samples(model, params, n.iter = n_iter, thin = 1000)
)

# Estimate running time for each iteration by dividing t evenly
t_vector <- cumsum(rep(t[3] / nrow(samples[[1]]), nrow(samples[[1]]))) / 60

# Save raw samples from each chain
for (i in 1:n_chains) {
  filename <- paste0('../data/param_est/samples_raw', i, '.csv')
  write.csv(samples[[i]], filename, row.names = FALSE)
}

# Turn raw samples data into a dataframe
samples_df <- data.frame()
for (i in 1:n_chains) {
  tmp <- data.frame(samples[[i]], iter = 1:nrow(samples[[i]]),
                           time = t_vector, chain = i) %>%
    gather(param, value, -iter, -time, -chain)
  samples_df <- rbind(samples_df, tmp)
}

# Calculate cumulative mean
samples_df <- samples_df %>%
  group_by(chain, param) %>%
  mutate(cum_mean = cumsum(value) / (iter))

# Plot mean vs iter to determine burin-in period
plt_dir <- '../plots/param_est_iter10m_thin1k/'
for (param in params) {
  samples_param <- samples_df %>%
    filter_(interp(~ param == p, p = param))
  
  plt <- ggplot(samples_param,
                aes(iter, cum_mean, linetype = as.factor(chain))) +
    geom_line() +
    theme_bw() +
    ggtitle(paste('Mean vs iteration of', param))
  print(plt)
  plt + ggsave(paste0(plt_dir, 'mean_v_iter_', param, '.png'))
}

# Discard burn-in samples and recalculate cumulative mean
discard_idx <- 250
samples_df <- samples_df %>%
  filter(iter > discard_idx) %>%
  group_by(chain, param) %>%
  mutate(cum_mean = cumsum(value) / (iter - discard_idx), cum_mode = NA)

# Calculate cumulative mode
step <- 10
for (chain in 1:n_chains) {
  for (param in params) {
    samples_param <- samples_df %>%
      filter_(interp(~ chain == i, i = chain)) %>%
      filter_(interp(~ param == p, p = param))
    samples_param_cum <- c()
    modes <- c()
    for (i in seq(step, nrow(samples_param), step)) {
      samples_param_cum <- c(samples_param_cum, samples_param$value[(i - step + 1):i])
      d <- density(samples_param_cum, bw = 'SJ')
      mode <- c(rep(NA, step - 1), d$x[which.max(d$y)])
      modes <- c(modes, mode)
    }
    samples_df[samples_df['param'] == param & samples_df['chain'] == chain, ]$cum_mode <- modes
  }
}

# Save samples_df to csv
write.csv(samples_df, '../data/param_est/samples_iter10m_thin1k.csv', row.names = FALSE)

# Preprocess samples_df for ggplot
est_df <- samples_df %>% na.omit() %>%
  gather(estimator, estimate, -iter, -time, -param, -chain, -value) %>%
  select(time, chain, param, estimator, estimate)
den_df <- samples_df %>% select(chain, param, value)

for (param in params) {
  est_param <- est_df %>%
    filter_(interp(~ param == p, p = param)) %>%
    mutate(diff = abs(estimate - true_params[param]))
  den_param <- den_df %>%
    filter_(interp(~ param == p, p = param))
  
  # Plot mean and mode vs running time
  est_plt <- ggplot(est_param,
                    aes(time, estimate, colour = estimator, linetype = as.factor(chain))) +
    geom_line() +
    theme_bw() +
    xlab('time in minutes') +
    ggtitle(paste('Estimate vs running time of', param))
  print(est_plt)
  est_plt + ggsave(paste0(plt_dir, 'est_', param, '_v_time1.png'))
  
  # Plot error vs running time
  error_plt <- ggplot(est_param,
                      aes(time, diff, colour = estimator, linetype = as.factor(chain))) +
    geom_line() +
    theme_bw() +
    xlab('time in minutes') +
    ggtitle(paste('Error vs running time of', param))
  print(error_plt)
  error_plt + ggsave(paste0(plt_dir, 'err_', param, '_v_time1.png'))
  
  # Plot density of samples
  d_plt <- ggplot(den_param, aes(value, linetype = as.factor(chain))) +
    geom_density(bw = 'SJ') +
    geom_vline(aes(xintercept = true_params[param]),
               color = 'red', linetype = 'dashed') +
    theme_bw() +
    ggtitle(paste('Density of', param))
  print(d_plt)
  d_plt + ggsave(paste0(plt_dir, 'density_', param, '1.png'))
}

summary <- samples_df %>%
  filter(iter == max(iter)) %>%
  select(chain, param, cum_mean, cum_mode) %>%
  mutate(error_mean = abs(cum_mean - true_params[param]),
         error_mode = abs(cum_mode - true_params[param]))
print(summary)
