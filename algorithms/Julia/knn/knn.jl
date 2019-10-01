module KNN
    struct KNNClassifier
        n_neighbors::Int
        similarity_function::Function
    end


    function predict_sample(x::Array)
        return []
    end

    function fit(classifier::KNNClassifier, data_x:: Matrix, data_y:: Array)
        if size(data_x)[1] < classifier.n_neighbors || size(data_y)[1] < classifier.n_neighbors
            error("Input size should by greater than n_neighbors")
        end

        if  size(data_x)[1] != size(data_y)[1]
            error("X and Y have different sizes")
        end

        data = [(x, y) for x=x_array, y=y_array]
        
        sort(data, by = x -> x[1])

    end

    macro fit(classifier::KNNClassifier)
        return quote
        end
    end

    export KNNClassifier, predict_sample
end
