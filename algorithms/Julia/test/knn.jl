using Test
include("../knn/knn.jl")
include("../utils/similarity_functions.jl")
using .KNN
using .SimilarityFunctions


function F(x::Int)
    return x + 3
end

classifier = KNNClassifier(3, euclidean_distance) 
x_train = [[1, 1], [3, 1], [1,2], [1,2]]
y_train = [0, 0, 0, 1]
x_sample = [1, 2]


@testset "Test KNN module" begin
    @test predict(classifier, x_train, y_train, x_sample) == 0
end
