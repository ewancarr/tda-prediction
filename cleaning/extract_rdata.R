# Title:        Convert RData object into feather format for use with Python
# Author:       Ewan Carr
# Started:      2021-05-25

library(feather)
library(here)

load(here("data", "GENDEP", "raw", "from_raquel",
          "larger_sample", "data793.RData"),
     verbose = TRUE)

write_feather(data,
              here("data", "GENDEP", "clean", "data793.feather"))
