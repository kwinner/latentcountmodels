library(ggplot2)
library(dplyr)
library(tidyr)

plot.params.default <- function() {
  params <- list()
  
  # variables for x (control), y (response) axes. should be an index (integer or name)
  params$x.var <- 1 # x axis variable
  params$y.var <- 2 # y axis variable
  
  # labels for x and y axes
  params$x.lab <- NULL
  params$y.lab <- NULL
  
  # limits for x and y axes
  params$x.lim <- NULL
  params$y.lim <- NULL
  
  # scales for x and y axes \in {linear, log2, log10, log}
  params$x.scale <- 'linear' 
  params$y.scale <- 'linear'
  
  # how to group results, can be the index of a key variable or a function to be applied to each entry to produce a key value
  params$group.by <- NULL
  
  # labels for each group in the legend
  params$group.labels <- NULL
  params$legend.title <- NULL
  
  # plot title, subtitle
  params$title    <- ''
  params$subtitle <- ''
  
  # a boolean function to exclude some results from plotting
  params.exclude <- function(x) {FALSE}
  
  return(params)
}

extend.results <- function(results) {
  results$Y <- sapply(results$y, function(y) sum(str2mat(y)))
  
  return(results)
}

merge.results <- function(results) {
  # split data by method
  results.rgd   <- results[which(results$method == 'pco_rgd'),]
  results.trunc <- results[which(results$method == 'pco_trunc:default'),]
  
  merged <- merge(results.rgd, results.trunc,
                  by = c('iota', 'y'),
                  suffixes = c('.rgd', '.trunc'))
  
  return(merged)
}

plot.trunc <- function(results) {
  params <- plot.params.default()
  
  params$x.var <- 'rho.gen'
  params$y.var <- 'nll'
  
  # params$y.lim <- c(0, 750)
  
  params$group.by     <- "method"
  params$group.labels <- c("rgd", "trunc")
  
  plot.experiment(results, params)
}

plot.experiment <- function(results,
                            params = plot.params.default()) {
  # g <- ggplot(results, aes_string(x = params$x.var, 
  #                                 y = params$y.var))
  g <- ggplot(results)
  
  g <- g + geom_line(aes_string(x=factor(params$x.var), y=params$y.var, color=params$group.by), position=position_dodge(1))
  # g <- g + geom_boxplot(aes_string(x=factor(params$x.var), y=params$y.var, fill=params$group.by), position=position_dodge(1))
  
  if (!is.null(params$x.lim) && !is.null(params$y.lim))
    g <- g + coord_cartesian(xlim=params$x.lim, ylim=params$y.lim)
  else if (!is.null(params$x.lim))
    g <- g + coord_cartesian(xlim=params$x.lim)
  else if (!is.null(params$y.lim))
    g <- g + coord_cartesian(ylim=params$y.lim)
  
  if (is.null(params$legend.title)) {
    # should select scale_fill, scale_colour, scale_shape automatically by plot type
    g <- g + scale_colour_discrete(name   = '',
                                   labels = params$group.labels)
    g <- g + theme(legend.title = element_blank())
  } else {
    g <- g + scale_colour_discrete(name   = params$legend.title,
                                   labels = params$group.labels)
  }
  
  plot(g)
}

plot.nll <- function(results) {
  results.merged <- merge.results(results)
  # results.merged <- results
  results.merged <- extend.results(results.merged)

  results.merged$improvement <- results.merged$nll.trunc - results.merged$nll.rgd
  
  results.merged$improvement[which(results.merged$improvement > 20000)] <- NaN  
  
  g <- ggplot(results.merged, aes(x=Y, y=improvement)) + 
       # geom_line() +
       geom_point() + coord_cartesian(ylim=c(-10,5))
  
  plot(g)
}

plot.rt <- function(results) {
  results.merged <- merge.results(results)
  results.merged <- extend.results(results.merged)
  
  g <- ggplot(results.merged, aes(x=rt.trunc, y=rt.rgd)) +
    # geom_line() +
    geom_point()

  plot(g)
}

plot.niter <- function(results) {
  results.merged <- merge.results(results)
  results.merged <- extend.results(results.merged)
  
  g <- ggplot(results.merged, aes(x=n.iters.trunc, y=n.iters.rgd)) +
    # geom_line() +
    geom_point()
  
  plot(g)
}

plot.rt.iter <- function(results) {
  results$rt.iter <- results$rt / results$n.iters
  
  results.merged <- merge.results(results)
  results.merged <- extend.results(results.merged)
  
  g <- ggplot(results.merged, aes(x=rt.iter.trunc, y=rt.iter.rgd)) +
    # geom_line() +
    geom_point()
  
  plot(g)
}