# Title:        Make figures/tables for TDA prediction paper
# Author:       Ewan Carr
# Started:      2023-06-13

library(tidyverse)
library(here)

# Import and tidy metrics from internal validation ----------------------------

f <- here("metrics.csv")
metrics <- read_csv(f)

names(metrics)[1:4] <- c("model", "sample", "max_week", "prs")

metrics <- metrics |>
  select(prs, model, sample, max_week,
         test_auc, test_sens30, test_spec30,
         test_sens40, test_spec40, test_npv) |>
  pivot_longer(cols = test_auc:test_npv)

clean <- metrics |>
  separate(value,
           into = c("p50", "p2", "p98"),
           sep = ",") |>
  mutate(across(c(p50, p2, p98), parse_number),
         sample = case_when(
           str_detect(sample, "escitalopram") ~ "A) Escitalopram",
           str_detect(sample, "nortriptyline") ~ "B) Nortriptyline",
           str_detect(sample, "both") ~ "C) Combined")) |>
  select(prs, model, sample, max_week, metric = name, p50, p2, p98)

# Choose thresholds for sensitivity and specificity
# A, Escitalopram = 40%
# B, Nortriptyline = 30%
# C, Combined = 40%

clean <- clean |>
  mutate(
    metric = case_when(
      str_starts(sample, "A|C") & metric == "test_sens40" ~ "sens",
      str_starts(sample, "B") & metric == "test_sens30" ~ "sens",
      str_starts(sample, "A|C") & metric == "test_spec40" ~ "spec",
      str_starts(sample, "B") & metric == "test_spec30" ~ "spec",
      .default = metric
      )
  )

clean <- clean |>
  drop_na(metric) |>
  mutate(metric = case_match(metric,
                             "test_auc" ~ "AUC",
                             "sens" ~ "Sensitivity",
                             "spec" ~ "Specificity",
                             "test_npv" ~ "NPV"),
         max_week = if_else(model == "1. Baseline only", 0, max_week)) |>
  drop_na(metric)

# Select the best-performing model for each sample/week -----------------------

plot_data <- clean |>
  group_by(prs, metric, sample, max_week) |>
  arrange(sample, metric, max_week, prs) |>
  mutate(best = rank(desc(p50), ties.method = "first") == 1)  |>
  filter(best)

plot_data$sample2 <- str_remove(plot_data$sample, "^[A-C]\\) ")


# Figure 1 --------------------------------------------------------------------

# Performance of the best-performing model for increasing weeks of data (0, 2,
# 4, 6) stratified by sample (nortriptyline, escitalopram, combined), without
# PRS variables.

labels_2dp <- \(x) sprintf("%.2f", x)

p <- plot_data |>
  filter(!prs) |>
  ggplot() +
  aes(x = max_week,
      y = p50,
      color = sample2) +
  geom_point() +
  geom_line() +
  facet_wrap(~ metric) +
  theme_minimal(base_family = "Times New Roman") +
  scale_color_brewer(type = "qual", palette = "Set2") +
  scale_y_continuous(labels = labels_2dp) +
  theme(axis.title.y = element_blank(),
        legend.text = element_text(size = 12),
        strip.text = element_text(size = 12)) +
  labs(x = "Weeks of repeated measures",
       color = "Sample")



ggsave(p,
       file = here("figures", "figure1.png"),
       width = 7,
       height = 4,
       dpi = 300,
       bg = "white")

# Mini table below Figure 1
plot_data |>
  filter(!prs) |>
  ungroup() |>
  select(sample2, metric, p50, max_week) |>
  mutate(p50 = sprintf("%.3f", p50)) |>
  pivot_wider(names_from = max_week, values_from = p50) |>
  writexl::write_xlsx(here("figures", "below_figure1.xlsx"))

# Table: Best results by sample/week ------------------------------------------

dp <- \(x) sprintf("%.3f", x)

plot_data |>
  ungroup() |>
  mutate(cell = str_glue("{dp(p50)}\n[{dp(p2)}, {dp(p98)}]")) |>
  select(prs, sample, max_week, metric, cell) |>
  pivot_wider(names_from = c(prs, metric), values_from = cell) |>
  arrange(max_week, sample) |>
  writexl::write_xlsx(here("tables", "best_performance.xlsx"))

# Table: Results for all methods ----------------------------------------------

perf <- clean |>
  mutate(cell = dp(p50),
         metric = factor(metric,
                         levels = c("AUC", "NPV",
                                    "Sensitivity",
                                    "Specificity"))) |>
  select(sample, model, max_week, prs, metric, cell) |>
  distinct(sample, model, max_week, prs, metric, .keep_all = TRUE)

perf <- perf |>
  pivot_wider(names_from = c(metric, prs),
              values_from = cell,
              names_sort = TRUE) |>
  arrange(sample, max_week, model)

perf |>
  writexl::write_xlsx(here("tables", "all_methods.xlsx"))

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



###############################################################################
####                                                                      #####
####                Data completeness in repeated measures                #####
####                                                                      #####
###############################################################################

f <- c("A", "B", "C", "replong")
df <- map(f, \(f) haven::read_dta(str_glue("data/stata/{f}.dta")))
names(df) <- f



df$A$index
df$B$index

dc <- df$replong |>
  pivot_longer(-subjectid) |>
  mutate(week = parse_number(str_extract(name, "[0-9]+$")),
         name = str_extract(name, "(^.*)_w[0-9]+$", group = 1)) |>
  filter(week %in% 0:12)

dc_a <- filter(dc, subjectid %in% df$A$index) |> mutate(sample = "A) Escitalopram")
dc_b <- filter(dc, subjectid %in% df$B$index) |> mutate(sample = "B) Nortriptyline")
dc_c <- filter(dc, subjectid %in% df$C$index) |> mutate(sample = "C) Combined")

dc <- bind_rows(dc_a, dc_b, dc_c) |>
  group_by(week, name, sample) |>
  summarise(pct = naniar::pct_complete(value))

p_missing <- dc |>
  ggplot() +
  aes(x = week,
      y = pct,
      fill = factor(week),
      color = factor(week)) +
  geom_hline(yintercept = 80, linetype = 2, alpha = 0.5) +
  geom_vline(xintercept = 6, linetype = 2, alpha = 0.5) +
  geom_boxplot(alpha = 0.2) +
  scale_x_continuous(breaks = seq(0, 12, 2)) +
  scale_y_continuous(limits = c(50, 100)) +
  facet_wrap(~ sample,
             ncol = 1) +
  labs(y = "Percentage complete",
       x = "Week") +
  theme_minimal(base_family = "Times New Roman") +
  theme(legend.position = "none")
p_missing

ggsave(p_missing,
       file = here("figures", "missing.png"),
       width = 7,
       height = 8,
       dpi = 300,
       bg = "white")

