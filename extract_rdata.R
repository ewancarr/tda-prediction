library(feather)
p <- "~/Sync/Work/Projects/Active/TDA/data/GENDEP/raw/from_raquel/larger_sample/data793.RData"
load(p, verbose = TRUE)

write_feather(data,
              "~/Sync/Work/Projects/Active/TDA/data/GENDEP/clean/data793.feather")
