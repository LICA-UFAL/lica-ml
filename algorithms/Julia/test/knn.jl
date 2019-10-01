using Test
include("../knn/knn.jl")
using .KNN

@test predict_sample([]) == []

