#' Simulating Warped Sinusoids
#'
#' 07/20/2019
library("ggplot2")
library("dplyr")
library("splines")

n <- 10000
l <- 200

t0 <- seq(0, 1, length.out = l)

## plot a few example random warpings
p <- 3
t1 <- matrix(nrow = n, ncol = l)
plot(t0, t0)
## plot(t0, sin(4 * pi * t0))
for (i in seq_len(n)) {
    dt1 <- abs(cbind(1, ns(t0, df = p - 1)) %*% rnorm(p))
    t1[i, ] <- cumsum(dt1)
    t1[i, ] <- t1[i, ] / max(t1[i, ])    ## points(t0, sin(4 * pi * t1[i, ]), col = colors()[sample(655, 1)])

    ## points(t0, t1[i, ], col = rgb(0, 0, 0, 0.2))
    ## points(t0, sin(4 * pi * t1[i, ]), col = colors()[sample(655, 1)])
}

z <- matrix(0, nrow = nrow(t1), ncol = ncol(t1))
for (i in seq_len(nrow(t1))) {
    t1[i, ] <- t1[i, ] * runif(1, .1, 10) * pi
    z[i, ] <- runif(1, 2, 5) * sin(t1[i, ]) + runif(1, -1, 1)
}


write.csv(t1, "../../data/sinusoid/times.csv", row.names = FALSE)
write.csv(z, "../../data/sinusoid/values.csv", row.names = FALSE)
