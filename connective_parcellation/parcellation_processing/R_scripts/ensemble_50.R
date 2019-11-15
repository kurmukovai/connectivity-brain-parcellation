library(RcppCNPy)
library(clue)
library(glue)

args = commandArgs(trailingOnly=TRUE)
# parcellation_level = args[1]
# resolution = args[2]
random_seed = strtoi(args[1])
parcellation_level = 3
resolution = 10

path = "/data01/ayagoz/sparse_32_concon_HCP/parcellations/"
folder = glue("{path}connectivity_parcellation_level{parcellation_level}/{resolution}/")
for (r in seq(random_seed, random_seed+20))
{
file_names = as.list(dir(path = folder, pattern="*.npy"))
file_names = lapply(file_names, function(x) paste0(folder, x))
set.seed(r)
file_names = sample(file_names)
file_names = file_names[1:50]
partitions = lapply(file_names, function(x) npyLoad(x, "integer"))
partitions_as_membership = lapply(partitions, function(x) as.cl_partition(x))

ensemble = cl_ensemble(list=partitions_as_membership)
ensemble_partition = cl_consensus(ensemble, method = "HE")
ensemble_vector = cl_class_ids(ensemble_partition)

saveto = glue("/home/kurmukov/subject_stability/HE/HE_3_10_{r}.npy")                                  
npySave(saveto, ensemble_vector, mode="w")
print(saveto)
}