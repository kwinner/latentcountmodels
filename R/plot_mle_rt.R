setwd('~/Projects/latentcountmodels/')
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

mode <- 4               # choose exepriment between 1 and 6 above
dir <- 'mle_bprop_rt'   # files should be in ../data/dir

# A few setups
name <- c('pois_bin', 'pois_pois', 'pois_nb', 'nb_bin', 'nb_pois', 'nb_nb')[mode]
title <- c('Poisson immigration, Bernoulli offspring',
           'Poisson immigration, Poisson offspring',
           'Poisson immigration, Geometric offspring',
           'NB immigration, Bernoulli offspring',
           'NB immigration, Poisson offspring',
           'NB immigration, Geometric offspring')[mode]

# Create out dir if doesn't exist
out_dir <- paste('plots', dir, sep = '/')
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

# Read data
in_dir <- paste('data', dir, name, sep = '/')
files <- list.files(in_dir, pattern = '\\.csv$')
df <- data.frame()
for (file in files) {
  df <- rbind(df, read.csv(paste(in_dir, file, sep = '/'), header = F)[, 1:3])
}
names(df) <- c('grad', 'K', 'runtime')
df <- df %>% mutate_at('grad', as.logical)

# Scatter + mean
ggplot(df, aes(K, runtime)) +
  geom_point(alpha = 0.5, size = 2, aes(colour = grad)) +
  stat_summary(fun.y = mean, geom = 'line', size = 2, aes(colour = grad)) +
  scale_colour_discrete(name="Gradient",
                        breaks=c(TRUE, FALSE),
                        labels=c("Exact", "Numerical")) +
  ylab('Running time (in seconds)') +
  theme_bw() +
  theme(legend.position = c(0.12, 0.86), legend.key.size = unit(1, "cm"),
        legend.text=element_text(size=22), text = element_text(size=22)) +
  ggtitle(title) +
  ggsave(paste0('plots/', dir, '/', name, '.png'),
         width = 10, height = 6)
