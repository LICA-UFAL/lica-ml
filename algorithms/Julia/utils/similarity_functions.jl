module SimilarityFunctions
   function euclidean_distance(first_instance, second_instance)
       if size(first_instance)[1] == 1
           first_instance = vec(first_instance...)
       end

       if size(second_instance)[1] == 1
           second_instance = vec(second_instance...)
       end

       return sum((first_instance - second_instance) .^ 2) .^ .5
   end

   export euclidean_distance
end
