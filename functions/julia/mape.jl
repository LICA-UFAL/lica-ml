function mean_absolute_percentage_error(y_test::Array, predict::Array)
    try
        summation = 0
        for i in length(y_test)
            summation += abs((y_test[i] - predict[i]) / y_test[i]) * 100
        end

        mean_error = summation / length(y_test)
        return mean_error
    catch e
        if any(x -> x==0, y_test) && any(x -> x==0, predict)
            summation = 0
            y_test = y_test + 1
            predict = predict + 1

            for i in length(y_test)
                summation += abs((y_test[i] - predict[i]) / y_test[i]) * 100
            end

            mean_error = summation / length(y_test)
            return mean_error
        elseif any(x -> x==0, y_test)
            summation = 0
            y_test = y_test + 1

            for i in length(y_test)
                summation += abs((y_test[i] - predict[i]) / y_test[i]) * 100
            end

            mean_error = summation / length(y_test)
            return mean_error
        elseif any(x -> x==0, predict)
            summation = 0
            predict = predict + 1

            for i in length(y_test)
                summation += abs((y_test[i] - predict[i]) / y_test[i]) * 100
            end

            mean_error = summation / length(y_test)
            return mean_error
end