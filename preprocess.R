library(geosphere)
library(dummies)
library(plyr)
library(foreach)
library(ggplot2)
library(discretization)
library(geosphere)

#-------------------------------
# define functions
#-------------------------------

# equal-width discretization
EqualWidth <- function(x,n){
  x <- HampelFilter(x, 5)
  x <- x$y
  v_min <- min(x)
  v_max <- max(x)
  interval <- (v_max - v_min) / n
  print(paste0("min:",v_min))
  print(paste0("max:",v_max))
  print(paste0("interval:",interval))
  
  bin <- floor((x-v_min)/interval) + 1
  member <- (x-v_min)%%interval / interval
  bin[bin==(n+1)] <- n
  
  mm <- matrix(0, length(x), n)
  encoding <- apply(mm, c(1, 2), function(x) 0)
  for(i in c(1:length(x))) {
      encoding[i,bin[i]] <- 1
  }
  encoding <- data.frame(encoding)
  
  stopifnot(sum(encoding)==nrow(encoding))
  
  encoding
}



# position coding
PositionCoding <- function(x,n){
  x <- HampelFilter(x, 5)
  x <- x$y
  v_min <- min(x)
  v_max <- max(x)
  interval <- (v_max - v_min)
  
  normalized <- (x-v_min)/interval

  mm <- matrix(0, length(x), n)
  encoding <- apply(mm, c(1, 2), function(x) 0)
  for(i in c(1:n)) {
    encoding[,i] <- dnorm(normalized, mean=interval/n*(i-1), sd=interval/n)
  }
  encoding <- data.frame(encoding)
  

  encoding
}

# nominal
# equal-width discretization
toNominal <- function(x,n){
  x <- HampelFilter(x, 5)
  x <- x$y
  v_min <- min(x)
  v_max <- max(x)
  interval <- (v_max - v_min) / n
  print(paste0("min:",v_min))
  print(paste0("max:",v_max))
  print(paste0("interval:",interval))
  
  bin <- floor((x-v_min)/interval)
  bin[bin==(n)] <- n-1
  
  encoding <- data.frame(bin)
  
  encoding
}


# median filter
HampelFilter <- function (x, k,t0=3){
  min_v <- quantile(x, 0)
  max_v <- quantile(x, 0.95)
  
  n <- length(x)
  y <- x
  ind <- c()
  
  for (i in (k + 1):(n-k)){
    if (i %% 10000 == 0) {print(i)}
    if (x[i]<max_v && x[i]>min_v){
      next
    }
    ind <- c(ind, i)
    y[i] <- median(x[(i - k):(i + k)])
    if (y[i] > max_v) {y[i] <- max_v}
    if (y[i] < min_v) {y[i] <- min_v}
    
    
  }
  
  list(y = y, ind = ind)
}

# median filter
HampelFilter2 <- function (x, k,t0=3){
  min_v <- quantile(x, 0.05)
  max_v <- quantile(x, 0.95)
  
  n <- length(x)
  y <- x
  ind <- c()
  
  for (i in (k + 1):(n-k)){
    
    stdev <- sd(x[(i - k):(i + k)])
    mdn <-  median(x[(i - k):(i + k)])
    if (abs(x[i]-mdn)<3*stdev){
      next
    }
    ind <- c(ind, i)
    y[i] <- median(x[(i - k):(i + k)])
    
    if (y[i] > max_v) {y[i] <- max_v}
    if (y[i] < min_v) {y[i] <- min_v}
    
    
  }
  
  list(y = y, ind = ind)
}

lag_apply <- function(x, n, callback){
  k = length(x);
  result = rep(0, k);
  for(i in 1 : (k - n + 1)){
    result[i] <- callback(x[i :  (i + n -1)]);
  }    
  return(result);
}

#-------------------------------
# data processing
#-------------------------------


# load original data
#data <- read.csv("mobility.csv") # mobility data

# select columns
data <- subset(data, select=c("MMSI", "speedmin", "avgspeed", "stdspeed", "COARSE_FIS"))

# select the four classes with their ids
# car, walk, bus, bike: 1,2,3,6
four_class_data <- data[data$COARSE_FIS %in% c(1,2,3,6),]

# get total length
len = length(four_class_data$LONGITUDE)
four_class_data$rowNum <- seq(len)

# find starting and end points for each vessel
mmsi_index_last <- aggregate(four_class_data$rowNum, by=list(four_class_data$MMSI), function(x) tail(x, 1))
mmsi_index_first <- aggregate(four_class_data$rowNum, by=list(four_class_data$MMSI), function(x) head(x, 1))

labels <- four_class_data$COARSE_FIS[-mmsi_index_last$x]
labels[labels==6] <- 0 # change bike to label 0
mmsis <- four_class_data$MMSI[-mmsi_index_last$x]

# extract features
speedmin <-four_class_data$speedmin[-mmsi_index_last$x]
avgspd<-four_class_data$avgspeed[-mmsi_index_last$x]
stdspd<-four_class_data$stdspeed[-mmsi_index_last$x]
time <- four_class_data$date[-mmsi_index_last$x]
time <- as.POSIXct(time)

# calculate encoding/discretization
# -------------------------------------------------------------------
speedmin_en <- EqualWidth(speedmin,50)
avgspd_en <- EqualWidth(avgspd,50)
stdspd_en <- EqualWidth(stdspd,50)


# bind features
# -------------------------------------------------------------------
context_features <- cbind(mmsis, speedmin_en, avgspd_en, stdspd_en, labels)

# write to csv file
# -------------------------------------------------------------------
write.table(point_features, file = "mobility_context_features.csv", sep = ",", row.names = FALSE, col.names = FALSE)