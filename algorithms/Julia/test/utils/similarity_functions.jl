using Test
using .SimilarityFunctions


first_instace = [[1, 1, 1]]
second_instance = [[3, 3, 3]]

@testset "Test similarity functions" begin
    @test isapprox(euclidean_distance(first_instace, second_instance), 3, atol=2)
end
