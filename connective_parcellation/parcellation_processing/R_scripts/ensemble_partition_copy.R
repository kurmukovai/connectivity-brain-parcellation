library(RcppCNPy)
library(clue)
library(glue)

args = commandArgs(trailingOnly=TRUE)
parcellation_level = 3
resolution = 10
random_seed = 10 

path = "/data01/ayagoz/sparse_32_concon_HCP/parcellations/"
folder = glue("{path}connectivity_parcellation_level{parcellation_level}/{resolution}/")
file_names = as.list(dir(path = folder, pattern="*.npy"))
file_names = lapply(file_names, function(x) paste0(folder, x))
set.seed(random_seed)
file_names = sample(file_names)
file_names = file_names[1:213]
partitions = lapply(file_names, function(x) npyLoad(x, "integer"))
partitions_as_membership = lapply(partitions, function(x) as.cl_partition(x))


ensemble = cl_ensemble(list=partitions_as_membership)
ensemble_partition = cl_consensus(ensemble, method = "HE")
ensemble_vector = cl_class_ids(ensemble_partition)

saveto ="/home/kurmukov/ensemble10_200.npy"                                  
npySave(saveto, ensemble_vector, mode="w")
print(saveto)
