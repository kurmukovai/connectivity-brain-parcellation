library(RcppCNPy)
library(clue)
library(glue)

args = commandArgs(trailingOnly=TRUE)
parcellation_level = args[1]
resolution = args[2]
random_seed = args[3]

path = "/data01/ayagoz/sparse_32_concon_HCP/parcellations/"
folder = glue("connectivity_parcellation_level{path}{parcellation_level}/{resolution}/")
print(folder)
file_names = as.list(dir(path = folder, pattern="*.npy"))
file_names = lapply(file_names, function(x) paste0(folder, x))
set.seed(random_seed)
file_names = sample(file_names)
partitions = lapply(file_names, function(x) npyLoad(x, "integer"))
partitions_as_membership = lapply(partitions, function(x) as.cl_partition(x))

print(length(partitions_as_membership))
