def mean_absolute_percentage_error(y_test, predict):
    """
    The MAPE is a metric for evaluate regression models.

    - y_test: The partition which use for test your model.
    - predict: An array which contains the predicted values.

    Returns a unique value which means the 
    mean absolute percentage error.
    """

    try:
        summation = 0
        
        for i in range(len(y_test)):
            summation += abs((y_test[i] - predict[i]) / y_test[i]) * 100
        
        mean_error = summation / len(y_test)
        return mean_error
    except Exception as e:
        if any(y_test) == 0 and any(predict) == 0:
            summation = 0
            y_test = y_test + 1
            predict = predict + 1
        
            for i in range(len(y_test)):
                summation += abs((y_test[i] - predict[i]) / y_test[i]) * 100
        
            mean_error = summation / len(y_test)
            return mean_error
        elif any(predict) == 0:
            summation = 0
            predict = predict + 1

            for i in range(len(y_test)):
                summation += abs((y_test[i] - predict[i]) / y_test[i]) * 100
            
            mean_error = summation / len(y_test)
            return mean_error
        elif any(y_test) == 0:
            summation = 0
            y_test = y_test + 1

            for i in range(len(y_test)):
                summation += abs((y_test[i] - predict[i]) / y_test[i]) * 100
            
            mean_error = summation / len(y_test)
            return mean_error
