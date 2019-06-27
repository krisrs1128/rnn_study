#' Inspect GRU Gatings
#'
#' Are warpings related to learned gatings?
library("dplyr")
library("ggplot2")
library("readr")
library("reshape2")

#' reshaping data
times <- read_csv("data/sinusoid/times.csv") %>%
    mutate(id = row_number())
activs <- read_tsv("data/sinusoid_inspect/activations_003.csv", col_names = F) %>%
    rename(time = X1, layer = X2, variable = X3, k = X4, value = X5) %>%
    mutate(
        time = as.integer(gsub("t", "", time)),
        layer = as.factor(gsub("l", "", layer)),
        k = as.factor(k)
    )

mtimes <- melt(times, id.vars = "id") %>%
    mutate(time = as.integer(gsub("V", "", variable)))

#' warping funs
ggplot(mtimes) +
    geom_line(
        aes(x = time, y = value, group = id),
        alpha = 0.1, size = 0.2
    )

#' look at some of the h values
activ_sub <- activs %>%
    filter(variable == "h")
ggplot(activ_sub) +
    geom_line(
        aes(x = time, y = value, col = k)
    ) +
    scale_color_brewer(palette = "Set2") +
    facet_wrap(~ layer)

#' look at some of the z values
#' seems like h doesn't change much for a while, but then it's allowed
#' but what's with the spike? why before the change, it's forced to be still?
activ_sub <- activs %>%
    filter(variable == "z")
ggplot(activ_sub) +
    geom_line(
        aes(x = time, y = value, col = k)
    ) +
    scale_color_brewer(palette = "Set2") +
    facet_grid(layer ~ .)

activ_sub <- activs %>%
    filter(variable == "n")
ggplot(activ_sub) +
    geom_line(
        aes(x = time, y = value, col = k)
    ) +
    scale_color_brewer(palette = "Set2") +
    facet_wrap(~ layer)

plot(as.numeric(times[4, 1:50]))
plot(as.numeric(sin(4 * pi * times[4, 1:50])))
