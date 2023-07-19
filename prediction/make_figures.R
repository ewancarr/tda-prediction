# Title:        Make figures/tables for TDA prediction paper
# Author:       Ewan Carr
# Started:      2023-06-13

library(tidyverse)
library(here)

# Import and tidy metrics from internal validation ----------------------------

metrics <- read_csv(here("metrics.csv"), skip = 3, col_names = FALSE) |>
  pivot_longer(-X1)
names(metrics) <- c("metric", "key", "value")

keys <- read_csv(here("metrics.csv"), n_max = 3, col_names = FALSE) |>
  select(-X1) |>
  t() |>
  as.data.frame() |>
  rownames_to_column()
names(keys) <- c("key", "model", "sample", "max_week")

required_metrics <- c("test_auc", "test_sens30", "test_spec30", 
                      "test_sens40", "test_spec40", "test_npv")

clean_results <- full_join(metrics, keys, join_by("key")) |>
  filter(metric %in% required_metrics) |>
  separate(value,
           into = c("p50", "p2", "p98"),
           sep = ",") |>
  mutate(across(c(p50, p2, p98), parse_number),
         sample = case_when(
           str_detect(sample, "escitalopram") ~ "A) Escitalopram",
           str_detect(sample, "nortriptyline") ~ "B) Nortriptyline",
           str_detect(sample, "both") ~ "C) Combined")) |>
  select(sample, model, max_week, metric, p50, p2, p98)

# Choose thresholds for sensitivity and specificity
# A, Escitalopram = 40%
# B, Nortriptyline = 30%
# C, Combined = 40%

clean_results[str_starts(clean_results$sample, "A|C") &
              clean_results$metric == "test_sens40", ]$metric <- "sens"
clean_results[str_starts(clean_results$sample, "B") &
              clean_results$metric == "test_sens30", ]$metric <- "sens"
clean_results[str_starts(clean_results$sample, "A|C") &
              clean_results$metric == "test_spec40", ]$metric <- "spec"
clean_results[str_starts(clean_results$sample, "B") &
              clean_results$metric == "test_spec30", ]$metric <- "spec"

clean_results <- clean_results |>
  mutate(metric = case_match(metric,
                             "test_auc" ~ "AUC",
                             "sens" ~ "Sensitivity",
                             "spec" ~ "Specificity",
                             "test_npv" ~ "NPV"),
         max_week = parse_number(max_week),
         max_week = if_else(model == "1. Baseline only", 0, max_week)) |>
  drop_na(metric)
         
# Select the best-performing model for each sample/week -----------------------

plot_data <- clean_results |>
  group_by(metric, sample, max_week) |>
  arrange(sample, metric, max_week) |>
  mutate(best = rank(desc(p50), ties.method = "first") == 1)  |>
  filter(best)

# Plot ------------------------------------------------------------------------

labels_2dp <- \(x) sprintf("%.2f", x)

p <- plot_data |>
  ggplot() +
  aes(x = max_week,
      y = p50,
      ymin = p2,
      ymax = p98,
      color = sample) +
  geom_line() +
  geom_pointrange() +
  facet_wrap(~ metric) +
  theme_minimal() +
  scale_color_brewer(type = "qual", palette = "Set2") +
  scale_y_continuous(labels = labels_2dp) +
  theme(axis.title.y = element_blank(),
        strip.text = element_text(size = 12)) +
  labs(x = "Weeks of repeated measures",
       color = "Sample")

ggsave(p,
       file = here("figures", "best_model.png"),
       width = 8,
       height = 7,
       dpi = 300,
       bg = "white")

# Table: Best results by sample/week ------------------------------------------

dp <- \(x) sprintf("%.2f", x)

plot_data |>
  ungroup() |>
  mutate(cell = str_glue("{dp(p50)} [{dp(p2)}, {dp(p98)}]")) |>
  select(sample, max_week, metric, cell) |>
  pivot_wider(names_from = metric, values_from = cell) |>
  write_csv(here("tables", "best_performance.csv"))

# Table: Results for all methods ----------------------------------------------

clean_results |>
  mutate(cell = str_glue("{dp(p50)} [{dp(p2)}, {dp(p98)}]")) |>
  select(sample, model, max_week, metric, cell) |>
  distinct(sample, model, max_week, metric, .keep_all = TRUE) |> 
  pivot_wider(names_from = metric, values_from = cell) |>
  write_csv(here("tables", "performance_across_methods.csv"))

# Figure: Results for all methods ---------------------------------------------

clean_results |>
  filter(metric == "AUC") |>
  ggplot() +
  aes(x = max_week,
      y = p50,
      ymin = p2,
      ymax = p98,
      color = model) +
  geom_line() +
  geom_pointrange() +
  facet_wrap(~ sample) +
  theme_minimal()
