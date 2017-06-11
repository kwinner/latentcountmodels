library(dplyr)
library(ggplot2)

"
Experiments:
1. Poisson arrival, binomial branching
2. Poisson arrival, Poisson branching
3. Poisson arrival, negative binomial branching
4. Negative binomial arrival, binomial branching
5. Negative binomial arrival, Poisson branching
6. Negative binomial arrival, negative binomial branching
"

mode <- 1        # choose exepriment between 1 and 6 above
dir <- 'mle_out2' # files should be in ../data/dir

# A few setups
name <- c('pois_bin', 'pois_pois', 'pois_nb', 'nb_bin', 'nb_pois', 'nb_nb')[mode]
other_params <- ifelse(mode < 4, '5.0_0.6', '10.0_0.67_0.6') # arrival and observation params
title <- c('Poisson arrival, Bernoulli offspring',
           'Poisson arrival, Poisson offspring',
           'Poisson arrival, Geometric offspring',
           'NB arrival, Bernoulli offspring',
           'NB arrival, Poisson offspring',
           'NB arrival, Geometric offspring')[mode]
if ((mode-1) %% 3) {
  thetas <- c(0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6)
} else {
  thetas <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
}
r_idx <- ifelse(mode < 4, 9, 10)

# Create out dir if doesn't exist
out_dir <- paste('../plots', dir, sep = '/')
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

# Read data
in_dir <- paste('../data', dir, name, other_params, sep = '/')
files <- list.files(in_dir, pattern = '\\.csv$')
df <- data.frame()
for (file in files) {
  # Read true param
  theta <- as.numeric(strsplit(gsub('.csv', '', file), '_')[[1]])
  
  if (theta %in% thetas) {
    # Read estimated param
    theta_hat <- read.csv(paste(in_dir, file, sep = '/'), header = F)[, r_idx] 
    
    tmp <- data.frame(theta = theta, theta_hat = theta_hat)
    df <- rbind(df, tmp)
  }
}

# Plot utils
true_line <- function(x) {x}
xlab_str <- 'R'
ylab_str <- bquote(hat(R))
true_legend_str <- 'True R'
mean_legend_str <- bquote(Mean ~ hat(R))

# Boxplot
ggplot(df, aes(theta, theta_hat, group = as.factor(theta))) +
  geom_boxplot(outlier.shape = NA, lwd=1) +
  stat_function(fun = true_line, size = 2, colour = '#009E73', aes(linetype = 'a')) +
  theme_bw() +
  xlab(xlab_str) +
  ylab(ylab_str) +
  scale_x_continuous(breaks = df$theta %>% unique()) +
  scale_linetype_manual(NULL, values = c('a' = 2),
                        labels = c(true_legend_str)) +
  ggtitle(title) +
  theme(legend.position = c(0.875, 0.1), legend.key.size = unit(1, "cm"),
        legend.text=element_text(size=30), text = element_text(size=30)) +
  guides(linetype = guide_legend(override.aes=list(size=3))) +
  ggsave(paste0('../plots/', dir, '/', name, '_box.png'),
         width = 10, height = 6, units = 'in')

# Scatter + mean
# ggplot(df, aes(theta, theta_hat)) +
#   stat_function(fun = true_line, size = 0.5, aes(colour = 'a', linetype = 'a')) +
#   stat_summary(fun.y = mean, geom = 'line', aes(colour = 'b', linetype = 'b')) +
#   geom_point(alpha = 0.5) +
#   theme_bw() +
#   xlab(xlab_str) +
#   ylab(ylab_str) +
#   scale_x_continuous(breaks = df$theta %>% unique()) +
#   scale_linetype_manual(NULL, values = c('a' = 2, 'b' = 1),
#                         labels = c(true_legend_str, mean_legend_str)) +
#   scale_colour_manual(NULL, values = c('a' = '#009E73', 'b' = '#E69F00'),
#                       labels = c(true_legend_str, mean_legend_str)) +
#   ggtitle(title) +
#   ggsave(paste0('../plots', dir, other_params, '/', name, '_scatter.png'))
