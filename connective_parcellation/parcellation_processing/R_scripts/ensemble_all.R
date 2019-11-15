# Compute ensemble parcellation for
# all sparsity levels for a single 
# partition resolution

library(RcppCNPy)
library(clue)
library(glue)

args = commandArgs(trailingOnly=TRUE)
parcellation_level = args[1]
path = "/data01/ayagoz/sparse_32_concon_HCP/parcellations/"
steps = c(10,20,30,40,50,60,70,80,90,100)

for (step in steps)
{
folder = glue("{path}connectivity_parcellation_level{parcellation_level}/{step}/")
file_names = as.list(dir(path = folder, pattern="*.npy"))
file_names = lapply(file_names, function(x) paste0(folder, x))
partitions = lapply(file_names, function(x) npyLoad(x, "integer"))
partitions_as_membership = lapply(partitions, function(x) as.cl_partition(x))


ensemble = cl_ensemble(list=partitions_as_membership)
ensemble_partition = cl_consensus(ensemble, method = "HE")
ensemble_vector = cl_class_ids(ensemble_partition)

saveto = glue("{path}ensemble_parcellation/connectivity_parcellation_level{parcellation_level}/{step}/ensemble_{parcellation_level}_{step}.npy")                                  
npySave(saveto, ensemble_vector, mode="w")
print(saveto)
}