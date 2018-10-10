library(unmarked)

load("/Users/kwinner/Work/Data/peckers.RData")

data <- subset(peckers, peckers$specid == 3380)

counts <- as.matrix(data[,9:44])
dates  <- as.matrix(data[,45:80])

dur<- data[,81:116]          # Survey duration
dur[dur=="NAA"]<- NA
dur<- as.matrix(dur)
dur<- matrix(as.numeric(dur),ncol=ncol(dur), byrow=FALSE)
dur<- dur/60

bad <- is.na(dates)                      # Missing surveys
counts[bad] <- NA                        # NA out data from missing surveys

# Standardise dates
mean.date <- mean(dates, na.rm = TRUE)
sd.date <- sd(dates, na.rm = TRUE)
dates <- (dates - mean.date)/sd.date

intensity<- dur/data$route

mean.dur<- mean(dur,na.rm=TRUE)
sd.dur<- sd(dur,na.rm=TRUE)
dur<- (dur - mean.dur)/sd.dur

mean.int<- mean(intensity,na.rm=TRUE)
sd.int<- sd(intensity,na.rm=TRUE)
intensity<- (intensity-mean.int)/sd.int


# put data into 3D arrays and summarize
C <- DATE <- DUR <- INT<- array(NA, dim = c(267, 3, 12)) # Counts and Dates
for(k in 1:12){                              # Put into 3D array
  C[,,k] <- counts[,(k*3-2):(3*k)]
  DATE[,,k] <- dates[,(k*3-2):(3*k)]
  DUR[,,k]<- dur[,(k*3-2):(3*k)] 
  INT[,,k]<- intensity[,(k*3-2):(3*k)]
}


# mean.C <- apply(C, c(1,3), mean, na.rm = TRUE) # Mean count per site and year
# annual.mean <- apply(mean.C, 2, mean, na.rm = TRUE)
# site.mean <- apply(mean.C, 1, mean, na.rm = TRUE)
# nsites.with.data <- apply(!is.na(mean.C), 2, sum, na.rm = TRUE)
# cat("N sites with data:", nsites.with.data, "\n")
# nsites.detected <- apply(mean.C>0, 2, sum, na.rm = TRUE)
# cat("N sites detected:", nsites.detected, "\n")
# cat("Observed occupancy prob.:", nsites.detected/nsites.with.data, "\n")
# cat("Observed mean count (year):", annual.mean, "\n")
# cat("Observed mean count (site):\n")   ;   summary(site.mean)

Cwide<- C[,,1]
date<- DATE[,,1]
dur<- DUR[,,1]
int<- INT[,,1]
for(i in 2:12){
  Cwide<- cbind(Cwide, C[,,i])
  date<- cbind(date, DATE[,,i])
  dur<- cbind(dur, DUR[,,i])
  int<- cbind(int, INT[,,i])
}

sitecovs<- data.frame( cbind(elev = as.numeric(scale(data$elev)), 
                             forest = as.numeric(scale(data$forest)),  ilength=1/data$route )
)
obscovs<- list(date=date, dur = dur, int=int)

umf <- unmarkedFramePCO(y = Cwide, numPrimary=12, siteCovs=sitecovs, obsCovs = obscovs)