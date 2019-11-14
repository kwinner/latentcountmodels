library(ggplot2)
library(reshape2)

mark.size <- 5
include.trunc <- TRUE
alpha <- 0.8
font.size <- 20
legend.font.size <- 16

gdfwd.shape <- 15
apgf.shape <- 16
trunc.shape <- 17

# gdfwd.shape <- 10
# apgf.shape <- 7

if(include.trunc) {
  shapes <- c(gdfwd.shape, trunc.shape, apgf.shape)
  labels <- c("gdual-fwd", "trunc", "apgffwd-s")
} else {
  shapes <- c(gdfwd.shape, apgf.shape)
  labels <- c("gdual-fwd", "apgffwd-s")
}

for(i.experiment in 1:n.experiments) {
  title <- titles[i.experiment]
  
  pdf(paste0(experiment.name, ' ', title, '.pdf'))
  if(include.trunc) {
    plot.df <- data.frame(x = eval.mags[[i.experiment]], fwd = nll.fwd[[i.experiment]], truncpco = nll.truncpco[[i.experiment]],  apgf = nll.apgf[[i.experiment]])
  } else {
    plot.df <- data.frame(x = eval.mags[[i.experiment]], fwd = nll.fwd[[i.experiment]], apgf = nll.apgf[[i.experiment]])
  }
  plot.df <- melt(plot.df, id.vars = "x", variable.name = "Method", value.name = "NLL")
  g <- ggplot(plot.df, aes(col = Method, shape = Method)) +
       labs(x = x_labs[i.experiment], y = "NLL", title = titles[i.experiment], colour = "Method") +
       theme(plot.title = element_text(hjust = 0.5),
             text = element_text(size=font.size),
             legend.text = element_text(size=legend.font.size, family="mono")) +
       geom_point(aes(x = x, y = NLL), size = mark.size, alpha = alpha) +
       scale_shape_manual(values = shapes, labels = labels) + 
       scale_color_hue(labels = labels)
  # plot(g)
  print(g)
  dev.off()
  
  pdf(paste0(experiment.name, ' ', titles.rt[i.experiment], '.pdf'))
  if(include.trunc) {
    plot.df <- data.frame(x = eval.mags[[i.experiment]], fwd = rt.fwd[[i.experiment]], truncpco = rt.truncpco[[i.experiment]],  apgf = rt.apgf[[i.experiment]])
  } else {
    plot.df <- data.frame(x = eval.mags[[i.experiment]], fwd = rt.fwd[[i.experiment]], apgf = rt.apgf[[i.experiment]])
  }
  plot.df <- melt(plot.df, id.vars = "x", variable.name = "Method", value.name = "rt")
  g <- ggplot(plot.df, aes(col = Method, shape = Method)) +
    labs(x = x_labs[i.experiment], y = "runtime (s)", title = titles.rt[i.experiment], colour = "Method") +
    theme(plot.title = element_text(hjust = 0.5),
          text = element_text(size=font.size),
          legend.text = element_text(size=legend.font.size, family="mono")) +
    geom_point(aes(x = x, y = rt), size = mark.size, alpha = alpha) +
    scale_shape_manual(values = shapes, labels = labels) + 
    scale_color_hue(labels = labels)
  # plot(g)
  print(g)
  dev.off()
  
  # if(include.trunc)
  #   plot.df <- data.frame(x = eval.mags[[i.experiment]], fwd = nll.fwd[[i.experiment]], truncpco = nll.truncpco[[i.experiment]],  apgf = nll.apgf[[i.experiment]])
  # else
  #   plot.df <- data.frame(x = eval.mags[[i.experiment]], fwd = nll.fwd[[i.experiment]], apgf = nll.apgf[[i.experiment]])
  # g <- ggplot(plot.df, aes()) + 
  #   labs(x = x_labs[i.experiment], y = "NLL", title = titles[i.experiment], colour = "Method") +
  #   theme(plot.title = element_text(hjust = 0.5)) +
  #   geom_point(aes(x = eval.mags[[i.experiment]], y = fwd,  col = "gdual-fwd"), shape = gdfwd.shape, size = mark.size, alpha = alpha)
  #   
  # if(include.trunc) {
  #   g <- g + geom_point(aes(x = eval.mags[[i.experiment]], y = truncpco,  col = "trunc"), shape = trunc.shape, size = mark.size, alpha = alpha) +
  #            geom_point(aes(x = eval.mags[[i.experiment]], y = apgf, col = "apgffwd-s"), shape = apgf.shape, size = mark.size, alpha = alpha)
  # } else {
  #   g <- g + geom_point(aes(x = eval.mags[[i.experiment]], y = apgf, col = "apgffwd-s"), shape = apgf.shape, size = mark.size, alpha = alpha)
  # }
  # 
  # if(include.trunc)
  #   g <- g + guides(col = guide_legend(override.aes = list(col=c(1,2,3),shape= c(gdfwd.shape, trunc.shape, apgf.shape))))
  # else
  #   g <- g + guides(col = guide_legend(override.aes = list(col=c(1,2),shape= c(gdfwd.shape, apgf.shape))))
  # plot(g)
  # print(g)
  # dev.off()
  
  # pdf(paste0(title, '_rt', '.pdf'))
  # plot.df <- data.frame(x = eval.mags[[i.experiment]], apgf = rt.apgf[[i.experiment]], fwd = rt.fwd[[i.experiment]], truncpco = rt.truncpco[[i.experiment]])
  # g <- ggplot(plot.df, aes()) + 
  #   labs(x = x_labs[i.experiment], y = "runtime (s)", title = titles.rt[i.experiment], colour = "Method") +
  #   theme(plot.title = element_text(hjust = 0.5)) +
  #   geom_point(aes(x = eval.mags[[i.experiment]], y = fwd,  col = "gdual-fwd"), shape = gdfwd.shape, size = mark.size, alpha = alpha)
  # 
  # if(include.trunc) {
  #   g <- g + geom_point(aes(x = eval.mags[[i.experiment]], y = truncpco,  col = "trunc"), shape = trunc.shape, size = mark.size, alpha = alpha) +
  #     geom_point(aes(x = eval.mags[[i.experiment]], y = apgf, col = "apgffwd-s"), shape = apgf.shape, size = mark.size, alpha = alpha)
  # } else {
  #   g <- g + geom_point(aes(x = eval.mags[[i.experiment]], y = apgf, col = "apgffwd-s"), shape = apgf.shape, size = mark.size, alpha = alpha)
  # }
  # 
  # if(include.trunc)
  #   # g <- g + guides(col = guide_legend(override.aes = list(col=c(1,2,3),shape= c(gdfwd.shape, trunc.shape, apgf.shape))))
  #   g <- g + scale_color_discrete(breaks=c("gdual-fwd", "trunc", "apgffwd-s"))
  # else
  #   # g <- g + guides(col = guide_legend(override.aes = list(col=c(1,2),shape= c(gdfwd.shape, apgf.shape))))
  #   g <- g + scale_color_discrete(breaks=c("gdual-fwd", "apgffwd-s"))
  # # plot(g)
  # print(g)
  # dev.off()
}