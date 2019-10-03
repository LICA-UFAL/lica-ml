module KNN
    struct KNNClassifier
        n_neighbors::Int
        similarity_function::Function
    end

    get_class(row) = row[2]

    function predict_sample(classifier::KNNClassifier, data::Array, sample::Array)
        data = sort(data, by = x -> classifier.similarity_function(x[1], sample))
        classes = get_class.(data[end - classifier.n_neighbors: end])
        unique_classes = unique(classes)
        freqs = map(i -> (i, count(x -> x == i , classes)), unique_classes) |> Dict
        return argmax(freqs)
    end

    function predict(classifier::KNNClassifier, data_x:: Array, data_y:: Array, sample::Array)
        if size(data_x)[1] < classifier.n_neighbors || size(data_y)[1] < classifier.n_neighbors
            error("Input size should by greater than n_neighbors")
        end

        if  size(data_x)[1] != size(data_y)[1]
            error("X and Y have different sizes")
        end

        data = map((x, y) -> (x, y), eachrow(data_x), data_y)
        return predict_sample(classifier, data, sample)
    end

    export KNNClassifier, predict
end
