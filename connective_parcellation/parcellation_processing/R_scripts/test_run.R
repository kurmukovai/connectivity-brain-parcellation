library(RcppCNPy)
library(clue)
library(glue)

start = Sys.time()
args = commandArgs(trailingOnly=TRUE)
parcellation_level = args[1]
resolution = args[2]
folder = glue("/data01/ayagoz/sparse_32_concon_HCP/parcellations/connectivity_parcellation_level{parcellation_level}/{resolution}/")

file_names = as.list(dir(path = folder, pattern="*.npy"))
file_names = lapply(file_names, function(x) paste0(folder, x))
print(length(file_names))
set.seed(1)
file_names = sample(file_names)
partitions = lapply(file_names, function(x) npyLoad(x, "integer"))
partitions_as_membership = lapply(partitions, function(x) as.cl_partition(x))
end = Sys.time()
print(end - start)

start = Sys.time()
ensemble = cl_ensemble(list=partitions_as_membership)
options("verbose"=TRUE)
ensemble_partition = cl_consensus(ensemble, method = "HE")
ensemble_vector = cl_class_ids(ensemble_partition)
npySave("/home/kurmukov/test_ensemble_426_1.npy", ensemble_vector, mode="w")
end = Sys.time()
print(end - start)
print(getOption("verbose"))
