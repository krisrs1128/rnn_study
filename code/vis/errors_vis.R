#' Packaging CSV into JSON
#'
library("readr")
library("ggplot2")
library("dplyr")
library("jsonlite")

progress <- read_csv(
    "~/Desktop/progress.csv",
    col_names = c("batch", "index", "y", "y_hat", "phase", "epoch")
) %>%
    group_by(epoch, phase) %>%
    mutate(
        loss = mean((y_hat - y) ^ 2),
        ix = 32 * batch + index,
        y_hat = round(y_hat, 3)
    ) %>%
    ungroup()

psub <- progress %>%
    filter(batch == 0, index == 0)

ggplot(psub) +
    geom_point(
        aes(
            x = epoch,
            y = loss,
            col = phase
        )
    )

ggplot(progress %>% filter(epoch == 10) %>% sample_n(1000)) +
    geom_point(
        aes(x = y, y = y_hat, col = phase),
        alpha = 0.2, size = 1,
        position = position_jitter(w = 0.2)
    ) +
    scale_y_continuous(limits = c(0, 6)) +
    scale_x_continuous(limits = c(0, 6))

keep_ix <- progress %>%
    filter(epoch == 0) %>%
    group_by(phase) %>%
    sample_n(300)

prog_sub <- progress %>%
    filter(ix %in% keep_ix$ix)

epoch_data <- prog_sub %>%
    select(epoch, phase, loss) %>%
    unique()

epoch_json <- list()
for (phase in unique(epoch_data$phase)) {
    epoch_json[[phase]] <- epoch_data %>%
        filter_(sprintf("phase == '%s'", phase)) %>%
        select(epoch, loss)
}

errors_json <- list()
for (epoch in seq(0, 29, by = 1)) {
    errors_json[[as.character(epoch)]] <- prog_sub %>%
        filter_(sprintf("epoch == %d", epoch)) %>%
        select(phase, y, y_hat, ix)
}

write_json(epoch_json, "epochs.json")
write_json(errors_json, "errors.json")

validation <- read_csv("../../data/sinusoid/validation/values.csv") %>%
    mutate(
        ix = row_number() - 1,
        phase = "validation"
    )

train <- read_csv("../../data/sinusoid/train/values.csv") %>%
    mutate(
        ix = row_number() - 1,
        phase = "train"
    )

curves <- rbind(train, validation)

curves <- prog_sub %>%
    filter(epoch == 0) %>%
    left_join(curves)
    ## select(ix, phase, starts_with("V"))

curves_json <- list()
for (i in seq_len(nrow(curves))) {
    curves_json[[i]] <- list(
        "phase" = curves[[i, "phase"]],
        "ix" = curves[[i, "ix"]]
    )
    cur_data <- curves[i, ] %>%
        select(starts_with("V")) %>%
        unlist()
    sub_ix = seq(1, length(cur_data), by = 2)
    curves_json[[i]]$data <- data_frame(
        i = sub_ix,
        v = round(cur_data[sub_ix], 4)
    )
}

write_json(curves_json, "curves.json", auto_unbox=TRUE)


curves %>%
    filter(ix == 596) %>%
    select(starts_with("V")) %>%
    unlist() %>%
    plot()
abline(h = 0)
abline(h = 1)

test = curves %>%
    filter(ix == 928, phase == "validation") %>%
    select(starts_with("V")) %>%
    unlist()

sum(diff(test <= 1 & test >= 0) == -1)
