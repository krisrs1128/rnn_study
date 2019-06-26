#' Simulating Warped Sinusoids
#'
#' 06/25/2019

library("ggplot2")
library("dplyr")
library("splines")

n <- 1000
l <- 50

t0 <- seq(0, 1, length.out = l)

## plot a few example random warpings
p <- 3
t1 <- matrix(nrow = n, ncol = l)
plot(t0, t0)
## plot(t0, sin(4 * pi * t0))
for (i in seq_len(n)) {
    dt1 <- abs(cbind(1, ns(t0, df = p - 1)) %*% rnorm(p))
    t1[i, ] <- cumsum(dt1)
    t1[i, ] <- t1[i, ] / max(t1[i, ])
    points(t0, t1[i, ], col = rgb(0, 0, 0, 0.2))
    ## points(t0, sin(4 * pi * t1[i, ]), col = colors()[sample(655, 1)])
}

## write these to file
write.csv(t1, "../../data/sinusoid/times.csv", row.names = FALSE)
write.csv(sin(4 * pi * t1), "../../data/sinusoid/values.csv", row.names = FALSE)

## look at some of the predictions
y <- read.csv("../../data/sinusoid/values.csv")
y_hat <- read.csv("../../data/sinusoid/y_hat.csv")

for (i in seq_len(100)) {
    plot(as.numeric(y[i, 2:50]))
    points(as.numeric(y_hat[i, ]), col = "red")
    Sys.sleep(0.5)
}
