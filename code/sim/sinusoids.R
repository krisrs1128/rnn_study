#' Simulating Warped Sinusoids
#'
#' 06/25/2019

library("ggplot2")
library("dplyr")
library("splines")

n <- 100
l <- 50

t0 <- seq(0, 1, length.out = l)

## plot a few example random warpings
p <- 3
n_test <- 1000
t1 <- matrix(nrow = n_test, ncol = l)
plot(t0, t0)
## plot(t0, sin(4 * pi * t0))
for (i in seq_len(n_test)) {
    dt1 <- abs(cbind(1, ns(t0, df = p - 1)) %*% rnorm(p))
    t1[i, ] <- cumsum(dt1)
    t1[i, ] <- t1[i, ] / max(t1[i, ])
    points(t0, t1[i, ], col = rgb(0, 0, 0, 0.2))
    ## points(t0, sin(4 * pi * t1[i, ]), col = colors()[sample(655, 1)])
}

## write these to file
write.csv(t1, "../../data/sinusoid/times.csv", row.names = FALSE)
write.csv(sin(4 * pi * t1), "../../data/sinusoid/values.csv", row.names = FALSE)
