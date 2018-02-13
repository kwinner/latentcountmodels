library(rjags)
library(dplyr)
library(tidyr)
library(lazyeval)
library(ggplot2)

# Create simulation data
lambda <- c(30, 200)         # avg initial abundance of (juveniles, adults)
delta <- c(0.5, 0.7)         # survival probabilities of (juveniles, adults)
rho <- c(0.4, 0.8)           # detection probabilities of (juveniles, adults)
gamma <- 0.1                 # birth rate
phi <- 0.3                   # juvenile -> adult transition probability

K <- 10
n <- matrix(rep(NA, K * 2), 2)
z <- matrix(rep(NA, K * 2), 2)
m <- rep(NA, K)
t <- rep(NA, K)
for (k in 1:K) {
  if (k == 1) {
    n[, k] <- rpois(2, lambda)
  } else {
    n[, k] <- c(m[k - 1] + z[1, k - 1] - t[k - 1], z[2, k - 1] + t[k - 1])
  }
  
  z[, k] <- rbinom(2, n[, k], delta)
  m[k] <- matrix(rpois(1, gamma * n[2, k]))
  t[k] <- matrix(rbinom(1, z[1, k], phi))
}
y <- matrix(rbinom(K * 2, n, rho), 2)       # observed counts

# Initial values
z0 <- y
#m0 <- y

# Run JAGS
model <- jags.model('zipkin.bug',
                    data = list('y' = y, 'K' = k, 'lambda' = lambda,
                                'delta' = delta, 'rho' = rho,
                                'gamma' = gamma, 'phi' = phi),
                    inits = list('z' = z0), # 'm' = m0
                    n.chains = 1,
                    n.adapt = 1000)

samples <- coda.samples(model, c('n'), n.iter = 10000, thin = 20)
summary(samples)

# Plot mean vs iter of n for each k = 1, 2, ..., K
samples_df <- data.frame(samples[[1]], iter = 1:nrow(samples[[1]]))
colnames(samples_df) <- gsub('\\.', '', colnames(samples_df)) # clean column names
samples_df <- samples_df %>%
  gather(var_k, value, -iter) %>%
  separate(var_k, into = c('var', 'stage', 'k'), sep = 1:2) %>%
  filter(var == 'n') %>%
  mutate(stage = ifelse(stage == 1, 'juvenile', 'adult')) %>%
  group_by(stage, k) %>%
  mutate(cum_mean = cumsum(value) / iter)
samples_df$k <- factor(samples_df$k, levels = 1:K)

# Juveniles
ggplot(samples_df %>% filter(stage == 'juvenile'), aes(iter, cum_mean, colour = k)) +
  geom_line() +
  theme_bw() +
  ggtitle('Mean vs iteration of juvenile n') +
  ggsave('../plots/zipkin/mean_vs_iter_juvenile.png')

# Adults
ggplot(samples_df %>% filter(stage == 'adult'), aes(iter, cum_mean, colour = k)) +
  geom_line() +
  theme_bw() +
  ggtitle('Mean vs iteration of adult n') +
  ggsave('../plots/zipkin/mean_vs_iter_adult.png')

# Discard burn-in
samples_df_new <- samples_df %>% filter(iter > 100)
summary <- samples_df_new %>% group_by(k) %>%
  summarise(mean = mean(value), sd = sd(value))
print(summary)

# Plot histogram of n_k, k = 1, 2, ..., K
for (stage in c('juvenile', 'adult')) {
  samples_stage <- samples_df_new %>% filter_(interp(~ stage == s, s = stage))
  for (k in 1:K) {
    samples_k <- samples_stage %>%
      filter_(interp(~ k == t_step, t_step = k))
    x_range <- min(samples_k$value):max(samples_k$value)
    ggplot(samples_k, aes(value)) +
      geom_bar() +
      geom_vline(aes(xintercept = mean(value)),
                 color = 'red', linetype = 'dashed') +
      scale_x_continuous(breaks = x_range) +
      theme_bw() +
      ggtitle(paste0('Distribution of ', stage, ' n_', k)) +
      ggsave(paste0('../plots/zipkin/hist_', stage, '_', k, '.png'))
  }
}