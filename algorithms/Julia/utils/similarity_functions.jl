module SimilarityFunctions
   function euclidean_distance(first_instace, second_instance)
       if length(size(first_instace)) == 1
           first_instace = vec(first_instace...)
       end
       return sum((first_instace - second_instance) .^ 2) .^ .5
   end

   export euclidean_distance
end
