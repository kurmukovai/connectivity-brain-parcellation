library(RcppCNPy)
library(clue)

folder = "/cobrain/groups/ml_group/data/concon_partitions/partitions_3_level/"
file_names = as.list(dir(path = folder, pattern="*.npy"))
file_names = lapply(file_names, function(x) paste0(folder, x))

partitions = lapply(file_names, function(x) npyLoad(x, "integer"))

partitions_as_membership = lapply(partitions, function(x) as.cl_partition(x))

ensemble = cl_ensemble(list=partitions_as_membership)
ensemble_partition = cl_consensus(ensemble, method = "HE")#method="soft/symdiff")
ensemble_vector = cl_class_ids(ensemble_partition)

npySave("/home/kurmukovai/partition_ensemble_3_level400.npy", ensemble_vector, mode="w")
