library(rjags)
library(dplyr)
library(tidyr)
library(lazyeval)
library(ggplot2)

K <- 10
n_iter <- 10000

# Create simulation data
lambda <- 10                 # rate of new arrivals
delta <- 0.5                 # survival probability
rho <- 0.8                   # detection probability

# m <- rpois(K, lambda)        # new arrivals
# n <- rep(NA, K)              # abundance
# z <- rep(NA, K - 1)          # survivors
# n[1] <- m[1]
# for (k in 2:K) {
#   z[k - 1] <- rbinom(1, n[k - 1], delta)
#   n[k] <- m[k] + z[k - 1]
#   
# }
# y <- rbinom(K, n, rho)       # observed counts

y <- c(10, 11, 7, 15, 13, 17, 13, 13, 18, 15)

# Initial values
#z0 <- y[1:K-1]
m0 <- y

# Run JAGS
model <- jags.model('dail_madsen.bug',
                   data = list('y' = y, 'K' = K, 'lambda' = lambda,
                               'delta' = delta, 'rho' = rho),
                   inits = list('m' = m0), # 'z' = z0
                   n.chains = 1,
                   n.adapt = 1000)

samples <- coda.samples(model, c('n', 'm', 'z'), n.iter = n_iter, thin = 20)
summary(samples)

# Compute likelihood of each sample
samples_df <- data.frame(samples[[1]], iter = 1:nrow(samples[[1]]))
colnames(samples_df) <- gsub('\\.', '', colnames(samples_df)) # clean column names
n_samples <- as.matrix(samples_df[,substr(colnames(samples_df), 1, 1)=="n"])[seq(1, nrow(n_samples), 10), ]
m_samples <- as.matrix(samples_df[,substr(colnames(samples_df), 1, 1)=="m"])[seq(1, nrow(m_samples), 10), ]
z_samples <- as.matrix(samples_df[,substr(colnames(samples_df), 1, 1)=="z"])[seq(1, nrow(z_samples), 10), ]
y_mat <- matrix(y, nrow = nrow(n_samples), ncol = K, byrow = TRUE)

joint_y <- apply(dbinom(y_mat, n_samples, rho, log = TRUE), 1, sum)
joint_m <- apply(dpois(n_samples, lambda, log = TRUE), 1, sum)
joint_z <- apply(dbinom(z_samples, n_samples[, 1:K-1], delta, log = TRUE), 1, sum)

likelihoods <- joint_y + joint_m + joint_z
likelihoods_df <- data.frame(iter = 1:nrow(n_samples), likelihood = likelihoods)

# Plot likelihood vs iter
ggplot(likelihoods_df, aes(iter * 10, likelihood)) +
  geom_line() +
  theme_bw() +
  ggtitle('Likelihood vs number of iterations') +
  ggsave('../plots/likelihood_vs_iter.png')

# Plot mean vs iter for each k = 1, 2, ..., K
samples_df <- samples_df %>%
  gather(var_k, value, -iter) %>%
  separate(var_k, into = c('var', 'k'), sep = 1) %>%
  group_by(var, k) %>%
  mutate(cum_mean = cumsum(value) / iter)
samples_df$k <- factor(samples_df$k, levels = 1:K)
for (var in c('n', 'm', 'z')) {
  samples_var <- samples_df %>%
    filter_(interp(~ var == v, v = var))
  ggplot(samples_var, aes(iter, cum_mean, colour = k)) +
    geom_line() +
    theme_bw() +
    ggtitle(paste('Mean vs number of iterations of', var)) +
    ggsave(paste0('../plots/mean_vs_iter_', var, '.png'))
}

# Discard burn-in
samples_df_new <- samples_df %>% filter(iter > 100)
summary <- samples_df_new %>% group_by(k) %>%
  summarise(mean = mean(value), sd = sd(value))
print(summary)

# Plot histogram of n_k, k = 1, 2, ..., K
samples_n <- samples_df_new %>%
  filter(var == 'n')
for (k in 1:K) {
  samples_nk <- samples_n %>%
    filter_(interp(~ k == t_step, t_step = k))
  x_range <- min(samples_nk$value):max(samples_nk$value)
  ggplot(samples_nk, aes(value)) +
    geom_bar() +
    geom_vline(aes(xintercept = mean(value)),
               color = 'red', linetype = 'dashed') +
    scale_x_continuous(breaks = x_range) +
    theme_bw() +
    ggtitle(paste('Distribution of n', k, sep = '_')) +
    ggsave(paste0('../plots/hist_', k, '.png'))
}
