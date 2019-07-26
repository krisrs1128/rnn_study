#' Simulating Warped Sinusoids
#'
#' 07/20/2019
library("ggplot2")
library("dplyr")
library("splines")

n <- 12500
l <- 200

t0 <- seq(0, 1, length.out = l)

## Generate random warpings
p <- 4
t1 <- matrix(nrow = n, ncol = l)
plot(t0, t0)
for (i in seq_len(n)) {
    dt1 <- abs(cbind(1, ns(t0, df = p - 1, knots = c(0.25, .5, .75))) %*% rnorm(p + 1))
    t1[i, ] <- cumsum(dt1)
    t1[i, ] <- t1[i, ] / max(t1[i, ])
}

z <- matrix(0, nrow = nrow(t1), ncol = ncol(t1))
for (i in seq_len(nrow(t1))) {
    t1[i, ] <- t1[i, ] * runif(1, .1, 6) * pi
    z[i, ] <- sample(c(1, -1), 1) * runif(1, 1, 5) * sin(t1[i, ]) + runif(1, -3, 3)
}

train_ix <- sample(1:n, .8 * n)
write.csv(t1[train_ix, ], "../../data/sinusoid/train/times.csv", row.names = FALSE)
write.csv(z[train_ix, ], "../../data/sinusoid/train/values.csv", row.names = FALSE)
write.csv(t1[-train_ix, ], "../../data/sinusoid/validation/times.csv", row.names = FALSE)
write.csv(z[-train_ix, ], "../../data/sinusoid/validation/values.csv", row.names = FALSE)
