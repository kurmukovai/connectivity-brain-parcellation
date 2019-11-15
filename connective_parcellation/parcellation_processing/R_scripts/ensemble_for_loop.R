library(RcppCNPy)
library(clue)
library(glue)

args = commandArgs(trailingOnly=TRUE)
parcellation_level = args[1]
resolution = args[2]
random_seed = strtoi(args[3])

path = "/data01/ayagoz/sparse_32_concon_HCP/parcellations/"
folder = glue("{path}connectivity_parcellation_level{parcellation_level}/{resolution}/")

steps = c(0,1,2)
for (step in steps)
{
file_names = as.list(dir(path = folder, pattern="*.npy"))
file_names = lapply(file_names, function(x) paste0(folder, x))
random_seed = random_seed + step
set.seed(random_seed)
file_names = sample(file_names)
partitions = lapply(file_names, function(x) npyLoad(x, "integer"))
partitions_as_membership = lapply(partitions, function(x) as.cl_partition(x))


ensemble = cl_ensemble(list=partitions_as_membership)
ensemble_partition = cl_consensus(ensemble, method = "HE")
ensemble_vector = cl_class_ids(ensemble_partition)

saveto = glue("{path}ensemble_parcellation/shuffle_ensemble_order3_100/ensemble{random_seed}.npy")                                  
npySave(saveto, ensemble_vector, mode="w")
print(saveto)
}