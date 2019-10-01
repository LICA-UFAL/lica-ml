module KNN
    struct KNNClassifier
        n_neighbors::Int
        similarity_function::Function
    end

    function predict_sample(classifier::KNNClassifier, sample::Array)
        return function _predict_sample(data::Array)
            data = sort(data, by = x -> classifier.similarity_function(x[1], sample))
            classes = map(x -> x[2], data[end - classifier.n_neighbors: end])
            unique_classes = unique(classes)
            freqs = map(i -> (i, count(x -> x == i , classes)), unique_classes) |> Dict
            return argmax(freqs)
        end
    end

    function predict(classifier::KNNClassifier, data_x:: Array, data_y:: Array, sample::Array)
        if size(data_x)[1] < classifier.n_neighbors || size(data_y)[1] < classifier.n_neighbors
            error("Input size should by greater than n_neighbors")
        end

        if  size(data_x)[1] != size(data_y)[1]
            error("X and Y have different sizes")
        end

        data = map((x, y) -> (x, y), eachrow(data_x), data_y)
        return predict_sample(classifier, sample)(data)
    end

   # macro predict(classifier::KNNClassifier, data_x::Array, data_y::Array, sample::Array)
   #     return quote
   #         predict($(classifier), $(data_x), $(data_y), $(sample))
   #     end
   # end

    export KNNClassifier, predict
end
