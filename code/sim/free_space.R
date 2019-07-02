
y <- read.csv("../../data/sinusoid/values.csv")
N <- ncol(y)

grid <- list()

ix <- 1
for (i in seq_len(N)) {
    for (j in seq_len(N)) {
        cur_dist <- data.frame(
            "t1" = i,
            "t2" = j,
            "dist" = (y[1, i] - y[3, j]) ^ 2
        )
        grid[[ix]] <- cur_dist
        ix <- ix + 1
    }
}

grid <- do.call(rbind, grid)

library("tidyverse")
ggplot(grid) +
    geom_tile(
        aes(x = t1, y = t2, fill = log(1 + dist))
    ) +
    coord_fixed()
